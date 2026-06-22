from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn.models import MLP

from model_building import _labels_to_index, build_LM_model
from runtime_env import _max_cuda_memory_allocated, _reset_cuda_peak_memory_stats, _resolve_device
from trainer_semantic_features import (
    _binary_prob_summary,
    _build_graph_attribute_features,
    _build_semantic_gate_node_attribute_features,
    _build_semantic_gate_features,
    _build_semantic_gate_local_competence_features,
    _build_text_attribute_features,
    _char_ratio_feature,
    _log1p_feature,
    _none_like_text,
    _parse_bool_feature,
    _resolve_gate_edge_tensors,
    _safe_float_feature,
    _standardize_attribute_features,
    _split_norm_user_text,
)
from trainer_semantic_models import _SemanticBreakRiskHead, _SemanticCorrectionDeferGate, _SemanticCorrectionGate
from utils import build_preparation_dir, capture_code_metadata, safe_torch_load, write_json, write_text, write_torch

_SEMANTIC_ANSWER_TEXT_BY_CLASS = {
    0: "No",
    1: "Yes",
}
_SEMANTIC_ANSWER_SLOT = "ASSISTANT_ANSWER:"


def _as_long_cpu_tensor(idx):
    if idx is None:
        return torch.empty(0, dtype=torch.long)
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().reshape(-1)
    return torch.tensor(idx, dtype=torch.long).reshape(-1)


def _score_all(labels, pred):
    labels_np = np.asarray(labels)
    pred_np = np.asarray(pred)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_np)),
        "macro_f1": float(f1_score(labels_np, pred_np, average="macro", zero_division=0)),
        "bot_f1": float(f1_score(labels_np, pred_np, average="binary", zero_division=0)),
        "count": int(labels_np.shape[0]),
    }


def _semantic_backbone_to_lm_model(args):
    backbone = str(getattr(args, "semantic_encoder", getattr(args, "semantic_backbone", "auto"))).lower()
    if backbone == "auto":
        return str(getattr(args, "text_encoder", getattr(args, "LM_model", "roberta"))).lower()
    mapping = {
        "roberta": "roberta",
        "roberta_finetuned": "roberta_finetuned",
        "qwen3_peft": "qwen3_peft",
    }
    if backbone not in mapping:
        raise ValueError(
            f"--experiment_task semantic_encoder_finetune does not train --semantic_encoder {backbone}. "
            "Use qwen3_peft for real Qwen LoRA PEFT or pass --embedding_path for frozen embeddings."
        )
    return mapping[backbone]


def _resolve_finetuned_roberta_checkpoint(args, seed):
    explicit = getattr(args, "finetuned_roberta_checkpoint_path", None)
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"--finetuned_roberta_checkpoint_path does not exist: {path}")
    dataset = str(getattr(args, "dataset", "TwiBot-20"))
    candidates = []
    for root in (Path.cwd(), Path.cwd().parent, Path("/root/workspace/LMbot")):
        candidates.append(root / f"{dataset}_seed_{int(seed)}" / "checkpoints" / "LM_pretrain" / "best.pkl")
        candidates.append(root / f"{dataset}_seed_{int(seed)}" / "checkpoints" / "LM" / "best.pkl")
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def _load_simteg_lm_checkpoint_into_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"SimTeG checkpoint at {checkpoint_path} does not contain a model state_dict.")
    lm_state = {
        key[len("LM.") :]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("LM.")
    }
    if not lm_state:
        raise ValueError(f"SimTeG checkpoint at {checkpoint_path} does not contain LM.* encoder weights.")
    load_result = model.load_state_dict(lm_state, strict=False)
    return {
        "loaded_key_count": int(len(lm_state)),
        "missing_key_count": int(len(getattr(load_result, "missing_keys", []))),
        "unexpected_key_count": int(len(getattr(load_result, "unexpected_keys", []))),
    }


def _resolve_routed_semantic_splits(args, data):
    routed_path = getattr(args, "routed_nodes_path", None)
    if not routed_path:
        return None
    path = Path(routed_path)
    if not path.exists():
        raise FileNotFoundError(f"--routed_nodes_path does not exist: {path}")
    if path.suffix == ".pt":
        raise ValueError("semantic routed-node split expects a JSON routed_nodes_path, not a .pt payload.")
    import json

    with open(path, "r", encoding="utf-8") as f:
        routed = json.load(f)
    node_ids = list(routed.get("node_ids") or [])
    split_counts = routed.get("split_counts") or {}
    train_count = int(split_counts.get("train", 0))
    valid_count = int(split_counts.get("valid", split_counts.get("dev", 0)))
    test_count = int(split_counts.get("test", 0))
    total = train_count + valid_count + test_count
    if total <= 0 or total > len(node_ids):
        raise ValueError(
            f"Invalid routed split_counts in {path}: train={train_count}, valid={valid_count}, test={test_count}, "
            f"node_ids={len(node_ids)}"
        )
    train_idx = torch.tensor(node_ids[:train_count], dtype=torch.long)
    valid_idx = torch.tensor(node_ids[train_count : train_count + valid_count], dtype=torch.long)
    test_idx = torch.tensor(node_ids[train_count + valid_count : train_count + valid_count + test_count], dtype=torch.long)
    labeled_count = int(data.get("labeled_node_count", len(data["user_text"])))
    for name, idx in {"train": train_idx, "valid": valid_idx, "test": test_idx}.items():
        if idx.numel() and (idx.min().item() < 0 or idx.max().item() >= labeled_count):
            raise ValueError(f"Routed {name} indices from {path} exceed labeled range [0, {labeled_count}).")
    return {
        "path": str(path),
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "split_counts": {
            "train": int(train_idx.numel()),
            "valid": int(valid_idx.numel()),
            "test": int(test_idx.numel()),
            "total": int(total),
        },
        "source_stage": routed.get("source_stage"),
        "selected_budget": routed.get("selected_budget"),
    }


def _resolve_semantic_split_indices(args, data, seed):
    routed_info = _resolve_routed_semantic_splits(args, data)
    if routed_info is not None:
        train_idx = _select_semantic_train_idx(
            routed_info["train_idx"],
            getattr(args, "semantic_train_limit", 0),
            seed,
        )
        valid_idx = _as_long_cpu_tensor(routed_info["valid_idx"])
        test_idx = _as_long_cpu_tensor(routed_info["test_idx"])
        scope = "routed_nodes_supervised_semantic_backbone"
        return train_idx, valid_idx, test_idx, routed_info, scope
    train_idx = _select_semantic_train_idx(
        data["train_idx"],
        getattr(args, "semantic_train_limit", 0),
        seed,
    )
    valid_idx = _as_long_cpu_tensor(data["valid_idx"])
    test_idx = _as_long_cpu_tensor(data["test_idx"])
    scope = "train_idx_supervised_semantic_backbone"
    return train_idx, valid_idx, test_idx, None, scope


def _semantic_command(args):
    flag_attr_pairs = [
        ("experiment_task", "experiment_task"),
        ("dataset", "dataset"),
        ("reset_split", "reset_split"),
        ("seeds", "seeds"),
        ("semantic_encoder", "semantic_encoder"),
        ("semantic_supervision_mode", "semantic_supervision_mode"),
        ("semantic_text_source_path", "semantic_text_source_path"),
        ("semantic_text_field", "semantic_text_field"),
        ("qwen_model_path", "qwen_model_path"),
        ("qwen_trust_remote_code", "qwen_trust_remote_code"),
        ("peft_rank", "peft_rank"),
        ("peft_alpha", "peft_alpha"),
        ("lm_batch_size", "batch_size_LM"),
        ("lm_learning_rate", "lr_LM"),
        ("lm_weight_decay", "weight_decay_LM"),
        ("dropout", "dropout"),
        ("lm_dropout", "LM_dropout"),
        ("lm_attention_dropout", "LM_att_dropout"),
        ("max_length", "max_length"),
        ("semantic_train_limit", "semantic_train_limit"),
        ("semantic_max_steps", "semantic_max_steps"),
        ("routed_nodes_path", "routed_nodes_path"),
        ("semantic_gate_base_outputs_path", "semantic_gate_base_outputs_path"),
        ("semantic_gate_candidate_output_paths", "semantic_gate_candidate_output_paths"),
        ("semantic_gate_candidate_names", "semantic_gate_candidate_names"),
        ("semantic_gate_epochs", "semantic_gate_epochs"),
        ("semantic_gate_hidden_dim", "semantic_gate_hidden_dim"),
        ("semantic_gate_learning_rate", "semantic_gate_learning_rate"),
        ("semantic_gate_weight_decay", "semantic_gate_weight_decay"),
        ("semantic_gate_break_weight", "semantic_gate_break_weight"),
        ("semantic_gate_threshold_policy", "semantic_gate_threshold_policy"),
        ("semantic_gate_feature_family", "semantic_gate_feature_family"),
        ("finetuned_roberta_checkpoint_path", "finetuned_roberta_checkpoint_path"),
        ("experiment_name", "experiment_name"),
        ("artifact_root", "artifact_root"),
        ("device", "device"),
        ("disable_wandb", "disable_wandb"),
    ]
    parts = ["python", "main.py"]
    for flag_name, attr_name in flag_attr_pairs:
        if not hasattr(args, attr_name):
            continue
        value = getattr(args, attr_name)
        if isinstance(value, bool):
            if value:
                parts.append(f"--{flag_name}")
            continue
        if value is not None:
            parts.extend([f"--{flag_name}", str(value)])
    return " ".join(parts)


def _semantic_tokenize(tokenizer, texts, max_length, device):
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    return {key: value.to(device) for key, value in tokenized.items()}


def _resolve_semantic_supervision_mode(args, lm_model):
    mode = str(getattr(args, "semantic_supervision_mode", "classifier") or "classifier").lower()
    if mode not in {"classifier", "answer_token"}:
        raise ValueError(f"Unsupported --semantic_supervision_mode: {mode}")
    if mode == "answer_token" and lm_model != "qwen3_peft":
        raise ValueError(
            "--semantic_supervision_mode answer_token currently requires "
            "--semantic_encoder qwen3_peft so the stage can train a causal-LM answer-token objective."
        )
    return mode


def _semantic_answer_text_from_label(label_idx):
    label_idx = int(label_idx)
    if label_idx not in _SEMANTIC_ANSWER_TEXT_BY_CLASS:
        raise ValueError(f"Unsupported semantic class for answer-token supervision: {label_idx}")
    return _SEMANTIC_ANSWER_TEXT_BY_CLASS[label_idx]


def _semantic_answer_token_mapping_manifest():
    return {
        "class_0": "No",
        "class_1": "Yes",
        "human": "No",
        "bot": "Yes",
    }


def _normalize_answer_slot_prompt(prompt):
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("answer-token supervision received an empty prompt.")
    if _SEMANTIC_ANSWER_SLOT not in text:
        raise ValueError(
            "answer-token supervision expects each prompt to end with a final "
            f"{_SEMANTIC_ANSWER_SLOT!r} slot. Use the DGP prompt sidecar 'prompt'/'full' field instead of raw user text."
        )
    prefix, _, suffix = text.rpartition(_SEMANTIC_ANSWER_SLOT)
    if suffix.strip():
        raise ValueError(
            "answer-token supervision expects prompts to stop at the final "
            f"{_SEMANTIC_ANSWER_SLOT!r} slot, but found a non-empty suffix after it."
        )
    prefix = prefix.rstrip()
    if not prefix:
        raise ValueError(
            f"answer-token supervision requires non-empty prompt content before {_SEMANTIC_ANSWER_SLOT!r}."
        )
    return f"{prefix}\n\n{_SEMANTIC_ANSWER_SLOT}"


def _normalize_answer_slot_prompts(prompts):
    return [_normalize_answer_slot_prompt(prompt) for prompt in prompts]


def _semantic_answer_token_sequences(tokenizer):
    answer_token_ids = {}
    for class_idx, answer_text in _SEMANTIC_ANSWER_TEXT_BY_CLASS.items():
        token_ids = tokenizer(" " + answer_text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer produced an empty sequence for answer text {answer_text!r}.")
        answer_token_ids[int(class_idx)] = list(token_ids)
    return answer_token_ids


def _pad_long_sequences(sequences, pad_value, device):
    if not sequences:
        raise ValueError("Expected at least one sequence to pad.")
    max_len = max(int(seq.numel()) for seq in sequences)
    padded = torch.full((len(sequences), max_len), int(pad_value), dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row_idx, seq in enumerate(sequences):
        seq = seq.long().reshape(-1)
        padded[row_idx, : seq.numel()] = seq
        attention[row_idx, : seq.numel()] = 1
    return padded.to(device), attention.to(device)


def _semantic_prompt_token_ids(tokenizer, prompts, *, max_length, reserved_answer_tokens):
    if int(max_length) <= int(reserved_answer_tokens):
        raise ValueError(
            f"--max_length={max_length} is too small for answer-token supervision with "
            f"reserved_answer_tokens={reserved_answer_tokens}."
        )
    prompts = _normalize_answer_slot_prompts(prompts)
    prompt_max_length = int(max_length) - int(reserved_answer_tokens)
    tokenized = tokenizer(
        list(prompts),
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_max_length,
        return_attention_mask=False,
    )
    return [torch.tensor(ids, dtype=torch.long) for ids in tokenized["input_ids"]]


def _build_answer_token_train_batch(tokenizer, prompts, labels, *, max_length, device, answer_token_ids):
    reserved_answer_tokens = max(len(ids) for ids in answer_token_ids.values())
    prompt_token_ids = _semantic_prompt_token_ids(
        tokenizer,
        prompts,
        max_length=max_length,
        reserved_answer_tokens=reserved_answer_tokens,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose pad_token_id for answer-token supervision.")
    input_sequences = []
    label_sequences = []
    for prompt_ids, label in zip(prompt_token_ids, labels):
        answer_ids = torch.tensor(answer_token_ids[int(label)], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        label_ids = torch.cat(
            [
                torch.full((prompt_ids.numel(),), -100, dtype=torch.long),
                answer_ids.clone(),
            ],
            dim=0,
        )
        input_sequences.append(input_ids)
        label_sequences.append(label_ids)
    input_ids, attention_mask = _pad_long_sequences(input_sequences, pad_token_id, device)
    labels_tensor, _ = _pad_long_sequences(label_sequences, -100, device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def _build_decoder_prompt_batch(tokenizer, prompts, *, max_length, device, answer_token_ids):
    reserved_answer_tokens = max(len(ids) for ids in answer_token_ids.values())
    prompt_token_ids = _semantic_prompt_token_ids(
        tokenizer,
        prompts,
        max_length=max_length,
        reserved_answer_tokens=reserved_answer_tokens,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must expose pad_token_id for decoder prompt batching.")
    input_ids, attention_mask = _pad_long_sequences(prompt_token_ids, pad_token_id, device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _decoder_last_token_embeddings(hidden_states, attention_mask):
    last_idx = attention_mask.sum(dim=1).clamp_min(1) - 1
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx].float()


def _score_answer_token_candidates(model, tokenizer, prompts, *, max_length, device, answer_token_ids):
    prompt_batch = _build_decoder_prompt_batch(
        tokenizer,
        prompts,
        max_length=max_length,
        device=device,
        answer_token_ids=answer_token_ids,
    )
    with torch.no_grad():
        prompt_out = model.LM(**prompt_batch, output_hidden_states=True)
    prompt_embeddings = _decoder_last_token_embeddings(prompt_out.hidden_states[-1], prompt_batch["attention_mask"]).detach().cpu()

    flat_sequences = []
    flat_labels = []
    flat_prompt_row = []
    flat_class_idx = []
    class_order = [0, 1]
    for prompt_row, prompt_ids in enumerate(prompt_batch["input_ids"].detach().cpu()):
        prompt_len = int(prompt_batch["attention_mask"][prompt_row].sum().item())
        prompt_ids = prompt_ids[:prompt_len]
        for class_idx in class_order:
            candidate_ids = torch.tensor(answer_token_ids[class_idx], dtype=torch.long)
            full_ids = torch.cat([prompt_ids, candidate_ids], dim=0)
            labels_ids = torch.cat(
                [
                    torch.full((prompt_ids.numel(),), -100, dtype=torch.long),
                    candidate_ids.clone(),
                ],
                dim=0,
            )
            flat_sequences.append(full_ids)
            flat_labels.append(labels_ids)
            flat_prompt_row.append(int(prompt_row))
            flat_class_idx.append(int(class_idx))

    pad_token_id = tokenizer.pad_token_id
    input_ids, attention_mask = _pad_long_sequences(flat_sequences, pad_token_id, device)
    labels_tensor, _ = _pad_long_sequences(flat_labels, -100, device)
    with torch.no_grad():
        scored_out = model.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        token_logits = scored_out.logits[:, :-1, :].float()
        target_tokens = input_ids[:, 1:]
        target_mask = labels_tensor[:, 1:] != -100
        token_log_prob = torch.log_softmax(token_logits, dim=-1)
        gathered = token_log_prob.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        gathered = gathered * target_mask.float()
        token_count = target_mask.sum(dim=1).clamp_min(1)
        candidate_scores = gathered.sum(dim=1) / token_count.float()

    logits = torch.zeros((len(prompts), 2), dtype=torch.float32)
    for score, prompt_row, class_idx in zip(
        candidate_scores.detach().cpu(),
        flat_prompt_row,
        flat_class_idx,
    ):
        logits[int(prompt_row), int(class_idx)] = float(score.item())
    return prompt_embeddings, logits


def _classification_metrics_from_logits(logits, labels, idx):
    idx = _as_long_cpu_tensor(idx)
    if idx.numel() == 0:
        return {"loss": float("inf"), "accuracy": 0.0, "macro_f1": 0.0, "bot_f1": 0.0, "count": 0}
    logits_idx = logits[idx]
    labels_idx = labels[idx]
    pred = logits_idx.argmax(dim=1)
    scores = _score_all(labels_idx.numpy(), pred.numpy())
    scores["loss"] = float(F.cross_entropy(logits_idx, labels_idx).item())
    return scores


def _semantic_eval_indices(train_idx, valid_idx, test_idx, total_count):
    pieces = [
        _as_long_cpu_tensor(train_idx),
        _as_long_cpu_tensor(valid_idx),
        _as_long_cpu_tensor(test_idx),
    ]
    pieces = [piece for piece in pieces if piece.numel()]
    if not pieces:
        return torch.arange(int(total_count), dtype=torch.long)
    return torch.unique(torch.cat(pieces).long(), sorted=True)


def _resolve_embedding_classifier_command(args):
    keys = [
        "experiment_task",
        "dataset",
        "reset_split",
        "seeds",
        "embedding_path",
        "semantic_encoder",
        "semantic_text_source_path",
        "semantic_text_field",
        "LM_classifier_n_layers",
        "LM_classifier_hidden_dim",
        "dropout",
        "lm_learning_rate",
        "lm_weight_decay",
        "semantic_train_limit",
        "routed_nodes_path",
        "experiment_name",
        "artifact_root",
        "device",
        "disable_wandb",
    ]
    parts = ["python", "main.py"]
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            continue
        if value is not None:
            parts.extend([f"--{key}", str(value)])
    return " ".join(parts)


def _read_jsonl_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_semantic_texts(args, data):
    base_texts = list(data["user_text"])
    source_path = getattr(args, "semantic_text_source_path", None)
    if not source_path:
        return base_texts, {
            "source": "data.user_text",
            "path": "",
            "field": "",
            "override_count": 0,
            "fallback_count": 0,
        }
    path = Path(source_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--semantic_text_source_path does not exist: {path}")
    requested_field = str(getattr(args, "semantic_text_field", "prompt") or "prompt")
    fallback_fields = [requested_field, "prompt", "user", "text", "full"]
    seen_fields = []
    override_count = 0
    fallback_count = 0
    for row in _read_jsonl_rows(path):
        if "node_id" not in row:
            continue
        node_id = int(row["node_id"])
        if node_id < 0 or node_id >= len(base_texts):
            continue
        text = ""
        used_field = ""
        for field in fallback_fields:
            if field in row and str(row.get(field) or "").strip():
                text = str(row[field]).strip()
                used_field = str(field)
                break
        if not text:
            fallback_count += 1
            continue
        base_texts[node_id] = text
        override_count += 1
        if used_field not in seen_fields:
            seen_fields.append(used_field)
    if override_count <= 0:
        raise ValueError(f"--semantic_text_source_path did not provide any usable node text rows: {path}")
    return base_texts, {
        "source": "semantic_text_source_path",
        "path": str(path),
        "field": requested_field,
        "used_fields": seen_fields,
        "override_count": int(override_count),
        "fallback_count": int(fallback_count),
        "row_alignment": "full_graph_user_text_with_node_id_overrides",
    }


def _resolve_cached_semantic_embeddings(args, data, seed):
    requested = getattr(args, "embedding_path", None)
    if requested:
        path = Path(requested)
        source = "embedding_path"
    else:
        from model_building import resolve_seed_aware_roberta_embedding_path

        path = resolve_seed_aware_roberta_embedding_path(data, seed=seed)
        source = "semantic_encoder_roberta_seed_aware_iter_-1"
    if not path.exists():
        raise FileNotFoundError(f"semantic_embedding_classifier could not find embeddings at {path}")
    payload = safe_torch_load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "embeddings" in payload and torch.is_tensor(payload["embeddings"]):
            payload = payload["embeddings"]
        else:
            raise ValueError("semantic_embedding_classifier expects a tensor or a payload with 'embeddings'.")
    if not torch.is_tensor(payload) or payload.dim() != 2:
        raise ValueError("semantic_embedding_classifier expects a 2-D embedding tensor.")
    return payload.detach().cpu().float(), path, source


def _split_csv_arg(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _semantic_outputs_prob_pred(payload, *, pred_key="pred", prob_key="prob", logits_key="logits", prefix=""):
    if not isinstance(payload, dict):
        raise ValueError("semantic correction gate expects outputs payloads to be dictionaries.")
    pred = payload.get(f"{prefix}{pred_key}") if prefix else payload.get(pred_key)
    prob = payload.get(f"{prefix}{prob_key}") if prefix else payload.get(prob_key)
    logits = payload.get(f"{prefix}{logits_key}") if prefix else payload.get(logits_key)
    if pred is None and prefix:
        pred = payload.get(pred_key)
    if prob is None and prefix:
        prob = payload.get(prob_key)
    if logits is None and prefix:
        logits = payload.get(logits_key)
    if prob is None:
        if logits is None:
            raise ValueError("outputs payload must contain prob or logits.")
        prob = torch.softmax(logits.detach().cpu().float(), dim=1)
    else:
        prob = prob.detach().cpu().float()
    if pred is None:
        pred = prob.argmax(dim=1)
    else:
        pred = pred.detach().cpu().long().reshape(-1)
    if prob.dim() != 2 or prob.shape[1] != 2:
        raise ValueError(f"semantic correction gate expects binary probabilities, got shape {tuple(prob.shape)}.")
    if int(pred.numel()) != int(prob.shape[0]):
        raise ValueError(f"pred/prob row mismatch: pred={int(pred.numel())}, prob={int(prob.shape[0])}.")
    return prob, pred


def _load_base_prob_pred(path):
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Base outputs at {path} must be a dictionary.")
    prob, pred = _semantic_outputs_prob_pred(
        payload,
        pred_key="base_pred",
        prob_key="base_prob",
        logits_key="base_logits",
    )
    return prob, pred, payload


def _load_candidate_prob_pred(path):
    payload = safe_torch_load(path, map_location="cpu")
    prob, pred = _semantic_outputs_prob_pred(payload)
    return prob, pred, payload


def _semantic_gate_rewards(labels, base_pred, candidate_preds, break_weight):
    labels = labels.detach().cpu().long().reshape(-1)
    base_pred = base_pred.detach().cpu().long().reshape(-1)
    base_correct = base_pred == labels
    rewards = []
    break_masks = []
    for cand_pred in candidate_preds:
        cand_pred = cand_pred.detach().cpu().long().reshape(-1)
        cand_correct = cand_pred == labels
        reward = torch.zeros_like(labels, dtype=torch.float32)
        reward[(~base_correct) & cand_correct] = 1.0
        reward[base_correct & (~cand_correct)] = -float(break_weight)
        rewards.append(reward)
        break_masks.append(base_correct & (~cand_correct))
    return torch.stack(rewards, dim=1), torch.stack(break_masks, dim=1), base_correct


def _scores_for_indices(labels, pred, idx):
    idx = _as_long_cpu_tensor(idx)
    return _score_all(labels[idx].numpy(), pred[idx].numpy()) if idx.numel() else {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "bot_f1": 0.0,
        "count": 0,
    }


def _semantic_gate_apply(base_pred, candidate_preds, gate_prob, idx, threshold):
    idx = _as_long_cpu_tensor(idx)
    final_pred = base_pred.clone()
    selected_action = torch.full((int(base_pred.numel()),), -1, dtype=torch.long)
    selected_score = torch.zeros((int(base_pred.numel()),), dtype=torch.float32)
    if idx.numel() == 0:
        return final_pred, selected_action, selected_score
    scores_idx = gate_prob[idx]
    max_scores, best_actions = scores_idx.max(dim=1)
    take = max_scores >= float(threshold)
    take_idx = idx[take]
    if take_idx.numel():
        best_take = best_actions[take]
        for action_idx, cand_pred in enumerate(candidate_preds):
            action_nodes = take_idx[best_take == action_idx]
            if action_nodes.numel():
                final_pred[action_nodes] = cand_pred[action_nodes]
                selected_action[action_nodes] = int(action_idx)
                selected_score[action_nodes] = max_scores[take][best_take == action_idx]
    skipped_idx = idx[~take]
    if skipped_idx.numel():
        selected_score[skipped_idx] = max_scores[~take]
    return final_pred, selected_action, selected_score


def _semantic_gate_delta(labels, base_pred, final_pred, idx):
    idx = _as_long_cpu_tensor(idx)
    if idx.numel() == 0:
        return {
            "fix": 0,
            "break": 0,
            "net": 0,
            "changed": 0,
            "base_wrong": 0,
            "base_correct": 0,
            "conditional_fix_rate_on_base_wrong": 0.0,
            "correct_node_break_rate": 0.0,
        }
    base_correct = base_pred[idx] == labels[idx]
    final_correct = final_pred[idx] == labels[idx]
    fix = int(((~base_correct) & final_correct).sum().item())
    broke = int((base_correct & (~final_correct)).sum().item())
    changed = int((base_pred[idx] != final_pred[idx]).sum().item())
    base_wrong = int((~base_correct).sum().item())
    base_correct_count = int(base_correct.sum().item())
    return {
        "fix": fix,
        "break": broke,
        "net": int(fix - broke),
        "changed": changed,
        "base_wrong": base_wrong,
        "base_correct": base_correct_count,
        "conditional_fix_rate_on_base_wrong": float(fix / max(base_wrong, 1)),
        "correct_node_break_rate": float(broke / max(base_correct_count, 1)),
    }


def _select_semantic_gate_threshold(labels, base_pred, candidate_preds, gate_prob, valid_idx):
    valid_idx = _as_long_cpu_tensor(valid_idx)
    if valid_idx.numel() == 0:
        return {"threshold": 1.0, "selection": "empty_valid"}
    candidates = torch.linspace(0.0, 1.0, steps=101).tolist()
    valid_scores = torch.unique(gate_prob[valid_idx].reshape(-1).detach().cpu()).tolist()
    candidates.extend(float(score) for score in valid_scores)
    candidates = sorted(set(float(max(0.0, min(1.0, value))) for value in candidates))
    best = None
    for threshold in candidates:
        final_pred, _, _ = _semantic_gate_apply(base_pred, candidate_preds, gate_prob, valid_idx, threshold)
        delta = _semantic_gate_delta(labels, base_pred, final_pred, valid_idx)
        scores = _scores_for_indices(labels, final_pred, valid_idx)
        row = {
            "threshold": float(threshold),
            **delta,
            "accuracy": float(scores["accuracy"]),
            "macro_f1": float(scores["macro_f1"]),
        }
        key = (row["net"], row["macro_f1"], -row["break"], row["fix"], -row["changed"])
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def _semantic_defer_targets(rewards):
    row_best, best_candidate = rewards.max(dim=1)
    targets = torch.zeros((int(rewards.shape[0]),), dtype=torch.long)
    positive = row_best > 0
    targets[positive] = best_candidate[positive].long() + 1
    return targets


def _semantic_defer_apply(
    base_pred,
    candidate_preds,
    action_prob,
    idx,
    accept_threshold,
    break_prob=None,
    break_threshold=None,
):
    idx = _as_long_cpu_tensor(idx)
    final_pred = base_pred.clone()
    selected_action = torch.full((int(base_pred.numel()),), -1, dtype=torch.long)
    selected_score = torch.zeros((int(base_pred.numel()),), dtype=torch.float32)
    if idx.numel() == 0:
        return final_pred, selected_action, selected_score
    node_action_prob = action_prob[idx]
    base_score = node_action_prob[:, 0]
    candidate_score = node_action_prob[:, 1:].clone()
    if break_prob is not None and break_threshold is not None:
        safe_mask = break_prob[idx] <= float(break_threshold)
        candidate_score = candidate_score.masked_fill(~safe_mask, -1.0)
    best_scores, best_actions = candidate_score.max(dim=1)
    take = (best_scores >= float(accept_threshold)) & (best_scores > base_score)
    take_idx = idx[take]
    if take_idx.numel():
        best_take = best_actions[take]
        for action_idx, cand_pred in enumerate(candidate_preds):
            action_nodes = take_idx[best_take == action_idx]
            if action_nodes.numel():
                final_pred[action_nodes] = cand_pred[action_nodes]
                selected_action[action_nodes] = int(action_idx)
                selected_score[action_nodes] = best_scores[take][best_take == action_idx]
    skipped_idx = idx[~take]
    if skipped_idx.numel():
        selected_score[skipped_idx] = best_scores[~take].clamp_min(0.0)
    return final_pred, selected_action, selected_score


def _select_semantic_defer_policy(
    labels,
    base_pred,
    candidate_preds,
    action_prob,
    valid_idx,
    break_prob=None,
    break_budget=-1.0,
):
    valid_idx = _as_long_cpu_tensor(valid_idx)
    if valid_idx.numel() == 0:
        return {"accept_threshold": 1.0, "break_threshold": None, "selection": "empty_valid"}
    accept_candidates = torch.linspace(0.0, 1.0, steps=51).tolist()
    valid_scores = action_prob[valid_idx, 1:].reshape(-1).detach().cpu()
    if valid_scores.numel():
        quantiles = torch.linspace(0.0, 1.0, steps=21)
        accept_candidates.extend(float(score) for score in torch.quantile(valid_scores, quantiles).tolist())
    accept_candidates = sorted(set(float(max(0.0, min(1.0, value))) for value in accept_candidates))
    if break_prob is None:
        break_candidates = [None]
    else:
        break_candidates = torch.linspace(0.0, 1.0, steps=51).tolist()
        valid_breaks = break_prob[valid_idx].reshape(-1).detach().cpu()
        if valid_breaks.numel():
            quantiles = torch.linspace(0.0, 1.0, steps=21)
            break_candidates.extend(float(score) for score in torch.quantile(valid_breaks, quantiles).tolist())
        break_candidates = sorted(set(float(max(0.0, min(1.0, value))) for value in break_candidates))
    best = None
    best_budgeted = None
    for accept_threshold in accept_candidates:
        for break_threshold in break_candidates:
            final_pred, _, _ = _semantic_defer_apply(
                base_pred,
                candidate_preds,
                action_prob,
                valid_idx,
                accept_threshold,
                break_prob=break_prob,
                break_threshold=break_threshold,
            )
            delta = _semantic_gate_delta(labels, base_pred, final_pred, valid_idx)
            if float(break_budget) >= 0.0 and delta["correct_node_break_rate"] > float(break_budget):
                budget_ok = False
            else:
                budget_ok = True
            scores = _scores_for_indices(labels, final_pred, valid_idx)
            row = {
                "accept_threshold": float(accept_threshold),
                "break_threshold": None if break_threshold is None else float(break_threshold),
                **delta,
                "accuracy": float(scores["accuracy"]),
                "macro_f1": float(scores["macro_f1"]),
                "break_budget": float(break_budget),
                "break_budget_satisfied": bool(budget_ok),
            }
            key = (row["net"], row["macro_f1"], -row["break"], row["fix"], -row["changed"])
            if best is None or key > best[0]:
                best = (key, row)
            if budget_ok and (best_budgeted is None or key > best_budgeted[0]):
                best_budgeted = (key, row)
    return (best_budgeted or best)[1]


def run_semantic_correction_gate_seed(args, seed, data, experiment_root, run):
    stage_dir = build_preparation_dir(experiment_root, "semantic_correction_gate")
    code_provenance = capture_code_metadata(Path(__file__).resolve().parents[1])
    base_path = Path(str(getattr(args, "semantic_gate_base_outputs_path", "") or ""))
    candidate_paths = [Path(item) for item in _split_csv_arg(getattr(args, "semantic_gate_candidate_output_paths", None))]
    candidate_names = _split_csv_arg(getattr(args, "semantic_gate_candidate_names", None))
    if not candidate_names:
        candidate_names = [path.parent.name or f"candidate_{idx}" for idx, path in enumerate(candidate_paths)]
    manifest = {
        "contract": "semantic_correction_gate_v1",
        "status": "started",
        "seed": int(seed),
        "canonical_task_name": "semantic_correction_gate",
        "dataset": getattr(args, "dataset", "unknown"),
        "dataset_path": str(data.get("dataset_path", "")),
        "training_scope": "routed_nodes_base_aware_correction_gate",
        "notes": (
            "Validation-locked base-aware keep/change gate over existing candidate semantic outputs. "
            "This stage trains utility selection only; it does not regenerate prompts or update the candidate LLM/MLP."
        ),
        "command": _semantic_command(args),
        "code_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
        "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
        "stage_visibility": "public",
        "artifact_namespace": "preparation/semantic_correction_gate",
        "invocation": {
            "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
            "resolved_task": "semantic_correction_gate",
        },
        "base_outputs_path": str(base_path),
        "candidate_output_paths": [str(path) for path in candidate_paths],
        "candidate_names": list(candidate_names),
        "break_weight": float(getattr(args, "semantic_gate_break_weight", 2.0)),
        "threshold_policy": str(getattr(args, "semantic_gate_threshold_policy", "global")),
        "feature_family": str(getattr(args, "semantic_gate_feature_family", "probability")),
        "selection_policy": str(getattr(args, "semantic_gate_selection_policy", "threshold")),
        "safety_policy": str(getattr(args, "semantic_gate_safety_policy", "none")),
        "break_budget": float(getattr(args, "semantic_gate_break_budget", -1.0)),
    }
    write_json(stage_dir / "manifest.json", manifest)
    write_text(stage_dir / "command.txt", manifest["command"] + "\n")

    try:
        if not base_path.exists():
            raise FileNotFoundError(f"--semantic_gate_base_outputs_path does not exist: {base_path}")
        if not candidate_paths:
            raise ValueError("--semantic_gate_candidate_output_paths must list at least one candidate outputs.pt.")
        if len(candidate_names) != len(candidate_paths):
            raise ValueError("--semantic_gate_candidate_names length must match --semantic_gate_candidate_output_paths.")
        for path in candidate_paths:
            if not path.exists():
                raise FileNotFoundError(f"candidate outputs path does not exist: {path}")

        device = _resolve_device(getattr(args, "device", -1))
        if device.type == "cuda":
            _reset_cuda_peak_memory_stats(device)
        labels = _labels_to_index(data["labels"]).cpu()
        canonical_test_idx = _as_long_cpu_tensor(data["test_idx"])
        train_idx, valid_idx, test_idx, routed_info, _ = _resolve_semantic_split_indices(args, data, seed)
        if routed_info is None:
            raise ValueError("semantic_correction_gate requires --routed_nodes_path.")
        base_prob, base_pred, _ = _load_base_prob_pred(base_path)
        candidate_probs = []
        candidate_preds = []
        for path in candidate_paths:
            prob, pred, _ = _load_candidate_prob_pred(path)
            candidate_probs.append(prob)
            candidate_preds.append(pred)
        row_count = int(labels.numel())
        tensors_to_check = [base_prob, base_pred, *candidate_probs, *candidate_preds]
        for tensor in tensors_to_check:
            if int(tensor.shape[0]) != row_count:
                raise ValueError(f"semantic_correction_gate row mismatch: expected {row_count}, got {int(tensor.shape[0])}.")

        feature_family = str(getattr(args, "semantic_gate_feature_family", "probability") or "probability").lower()
        node_attribute_features = None
        node_attribute_metadata = None
        if feature_family in {"node_attribute", "local_competence"}:
            node_attribute_features, node_attribute_names, node_attribute_metadata = _build_semantic_gate_node_attribute_features(
                data,
                train_idx,
                row_count,
            )
            node_attribute_metadata["feature_names"] = list(node_attribute_names)
        elif feature_family == "probability":
            pass
        else:
            raise ValueError(f"Unsupported semantic_gate_feature_family={feature_family!r}")
        rewards, break_masks, base_correct = _semantic_gate_rewards(
            labels,
            base_pred,
            candidate_preds,
            float(getattr(args, "semantic_gate_break_weight", 2.0)),
        )
        local_competence_features = None
        local_competence_metadata = None
        descriptor_features = _build_semantic_gate_features(
            base_prob,
            base_pred,
            candidate_probs,
            candidate_preds,
            node_attribute_features=node_attribute_features,
        )
        if feature_family == "local_competence":
            local_competence_features, local_competence_metadata = _build_semantic_gate_local_competence_features(
                descriptor_features,
                train_idx,
                labels,
                base_pred,
                candidate_preds,
                rewards,
                break_masks,
                float(getattr(args, "semantic_gate_break_weight", 2.0)),
                k=int(getattr(args, "semantic_gate_local_k", 25)),
            )
            local_competence_metadata["descriptor_feature_dim"] = int(descriptor_features.shape[-1])
            local_competence_metadata["candidate_names"] = list(candidate_names)
            local_competence_metadata["node_attribute_descriptor_enabled"] = node_attribute_features is not None
        if feature_family == "local_competence" and node_attribute_metadata is not None:
            node_attribute_metadata["usage"] = "descriptor_and_gate_input"
        features = _build_semantic_gate_features(
            base_prob,
            base_pred,
            candidate_probs,
            candidate_preds,
            node_attribute_features=node_attribute_features,
            action_local_features=local_competence_features,
        )
        selection_policy = str(getattr(args, "semantic_gate_selection_policy", "threshold") or "threshold").lower()
        safety_policy = str(getattr(args, "semantic_gate_safety_policy", "none") or "none").lower()
        if selection_policy not in {"threshold", "defer_softmax"}:
            raise ValueError(f"Unsupported semantic_gate_selection_policy={selection_policy!r}")
        if safety_policy not in {"none", "break_first"}:
            raise ValueError(f"Unsupported semantic_gate_safety_policy={safety_policy!r}")
        if selection_policy != "defer_softmax" and safety_policy != "none":
            raise ValueError("--semantic_gate_safety_policy break_first requires --semantic_gate_selection_policy defer_softmax.")
        targets = (rewards > 0).float()
        sample_weights = torch.ones_like(targets)
        sample_weights[break_masks] = float(getattr(args, "semantic_gate_break_weight", 2.0))
        train_features = features[train_idx].to(device)
        epochs = max(int(getattr(args, "semantic_gate_epochs", 200)), 1)
        losses = []
        best_state = None
        best_break_state = None
        best_valid = None
        action_logits = None
        action_prob = None
        break_logits = None
        break_prob = None
        break_model = None
        break_threshold = None
        defer_targets = None
        if selection_policy == "threshold":
            train_targets = targets[train_idx].to(device)
            train_weights = sample_weights[train_idx].to(device)
            model = _SemanticCorrectionGate(
                in_channels=int(features.shape[-1]),
                hidden_channels=int(getattr(args, "semantic_gate_hidden_dim", 64)),
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(args, "semantic_gate_learning_rate", 1e-3)),
                weight_decay=float(getattr(args, "semantic_gate_weight_decay", 1e-4)),
            )
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits_train = model(train_features)
                loss_raw = F.binary_cross_entropy_with_logits(logits_train, train_targets, reduction="none")
                loss = (loss_raw * train_weights).sum() / train_weights.sum().clamp_min(1.0)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                run.log({"semantic_correction_gate_loss": loss_value, "semantic_correction_gate_epoch": epoch + 1})
                if epoch == epochs - 1 or epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        gate_prob_epoch = torch.sigmoid(model(features.to(device))).cpu()
                    threshold_row = _select_semantic_gate_threshold(labels, base_pred, candidate_preds, gate_prob_epoch, valid_idx)
                    key = (int(threshold_row["net"]), float(threshold_row["macro_f1"]), -int(threshold_row["break"]))
                    if best_valid is None or key > best_valid[0]:
                        best_valid = (key, dict(threshold_row))
                        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if best_state is None:
                raise RuntimeError("semantic_correction_gate failed to record a best state.")
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                gate_logits = model(features.to(device)).cpu()
            gate_prob = torch.sigmoid(gate_logits)
            threshold_row = _select_semantic_gate_threshold(labels, base_pred, candidate_preds, gate_prob, valid_idx)
            threshold = float(threshold_row["threshold"])
        else:
            defer_targets = _semantic_defer_targets(rewards)
            train_targets = defer_targets[train_idx].to(device)
            train_weights = torch.ones_like(train_targets, dtype=torch.float32)
            train_break_any = break_masks[train_idx].any(dim=1).to(device)
            train_weights[train_break_any] = float(getattr(args, "semantic_gate_break_weight", 2.0))
            model = _SemanticCorrectionDeferGate(
                in_channels=int(features.shape[-1]),
                hidden_channels=int(getattr(args, "semantic_gate_hidden_dim", 64)),
            ).to(device)
            if safety_policy == "break_first":
                break_model = _SemanticBreakRiskHead(
                    in_channels=int(features.shape[-1]),
                    hidden_channels=int(getattr(args, "semantic_gate_hidden_dim", 64)),
                ).to(device)
                train_break_targets = break_masks[train_idx].float().to(device)
                break_train_weights = torch.ones_like(train_break_targets)
                break_train_weights[train_break_targets > 0] = float(getattr(args, "semantic_gate_break_weight", 2.0))
                params = list(model.parameters()) + list(break_model.parameters())
            else:
                train_break_targets = None
                break_train_weights = None
                params = model.parameters()
            optimizer = torch.optim.AdamW(
                params,
                lr=float(getattr(args, "semantic_gate_learning_rate", 1e-3)),
                weight_decay=float(getattr(args, "semantic_gate_weight_decay", 1e-4)),
            )
            for epoch in range(epochs):
                model.train()
                if break_model is not None:
                    break_model.train()
                optimizer.zero_grad(set_to_none=True)
                logits_train = model(train_features)
                loss_raw = F.cross_entropy(logits_train, train_targets, reduction="none")
                loss = (loss_raw * train_weights).sum() / train_weights.sum().clamp_min(1.0)
                if break_model is not None:
                    break_logits_train = break_model(train_features)
                    break_loss_raw = F.binary_cross_entropy_with_logits(
                        break_logits_train,
                        train_break_targets,
                        reduction="none",
                    )
                    break_loss = (break_loss_raw * break_train_weights).sum() / break_train_weights.sum().clamp_min(1.0)
                    loss = loss + break_loss
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                run.log({"semantic_correction_gate_loss": loss_value, "semantic_correction_gate_epoch": epoch + 1})
                if epoch == epochs - 1 or epoch % 10 == 0:
                    model.eval()
                    if break_model is not None:
                        break_model.eval()
                    with torch.no_grad():
                        action_prob_epoch = F.softmax(model(features.to(device)), dim=1).cpu()
                        break_prob_epoch = torch.sigmoid(break_model(features.to(device))).cpu() if break_model is not None else None
                    threshold_row = _select_semantic_defer_policy(
                        labels,
                        base_pred,
                        candidate_preds,
                        action_prob_epoch,
                        valid_idx,
                        break_prob=break_prob_epoch,
                        break_budget=float(getattr(args, "semantic_gate_break_budget", -1.0)),
                    )
                    key = (int(threshold_row["net"]), float(threshold_row["macro_f1"]), -int(threshold_row["break"]))
                    if best_valid is None or key > best_valid[0]:
                        best_valid = (key, dict(threshold_row))
                        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                        if break_model is not None:
                            best_break_state = {name: value.detach().cpu().clone() for name, value in break_model.state_dict().items()}
            if best_state is None:
                raise RuntimeError("semantic_correction_gate failed to record a best state.")
            model.load_state_dict(best_state)
            model.eval()
            if break_model is not None:
                break_model.load_state_dict(best_break_state)
                break_model.eval()
            with torch.no_grad():
                action_logits = model(features.to(device)).cpu()
                action_prob = F.softmax(action_logits, dim=1)
                if break_model is not None:
                    break_logits = break_model(features.to(device)).cpu()
                    break_prob = torch.sigmoid(break_logits)
            gate_logits = action_logits[:, 1:]
            gate_prob = action_prob[:, 1:]
            threshold_row = _select_semantic_defer_policy(
                labels,
                base_pred,
                candidate_preds,
                action_prob,
                valid_idx,
                break_prob=break_prob,
                break_budget=float(getattr(args, "semantic_gate_break_budget", -1.0)),
            )
            threshold = float(threshold_row["accept_threshold"])
            break_threshold = threshold_row.get("break_threshold")
        final_pred = base_pred.clone()
        selected_action = torch.full((row_count,), -1, dtype=torch.long)
        selected_score = torch.zeros((row_count,), dtype=torch.float32)
        for split_idx in (train_idx, valid_idx, test_idx):
            if selection_policy == "threshold":
                split_final, split_action, split_score = _semantic_gate_apply(
                    base_pred,
                    candidate_preds,
                    gate_prob,
                    split_idx,
                    threshold,
                )
            else:
                split_final, split_action, split_score = _semantic_defer_apply(
                    base_pred,
                    candidate_preds,
                    action_prob,
                    split_idx,
                    threshold,
                    break_prob=break_prob,
                    break_threshold=break_threshold,
                )
            split_idx = _as_long_cpu_tensor(split_idx)
            final_pred[split_idx] = split_final[split_idx]
            selected_action[split_idx] = split_action[split_idx]
            selected_score[split_idx] = split_score[split_idx]

        metrics = {
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "validation_threshold": threshold_row,
            "feature_family": feature_family,
            "selection_policy": selection_policy,
            "safety_policy": safety_policy,
            "break_budget": float(getattr(args, "semantic_gate_break_budget", -1.0)),
            "node_attribute_feature_dim": int(node_attribute_features.shape[1]) if node_attribute_features is not None else 0,
            "local_competence_feature_dim": int(local_competence_features.shape[-1]) if local_competence_features is not None else 0,
            "local_competence_metadata": local_competence_metadata,
            "train": {
                "base": _scores_for_indices(labels, base_pred, train_idx),
                "gated": _scores_for_indices(labels, final_pred, train_idx),
                "delta": _semantic_gate_delta(labels, base_pred, final_pred, train_idx),
            },
            "validation": {
                "base": _scores_for_indices(labels, base_pred, valid_idx),
                "gated": _scores_for_indices(labels, final_pred, valid_idx),
                "delta": _semantic_gate_delta(labels, base_pred, final_pred, valid_idx),
            },
            "test": {
                "base": _scores_for_indices(labels, base_pred, test_idx),
                "gated": _scores_for_indices(labels, final_pred, test_idx),
                "delta": _semantic_gate_delta(labels, base_pred, final_pred, test_idx),
            },
            "canonical_full_test": {
                "base": _scores_for_indices(labels, base_pred, canonical_test_idx),
                "gated": _scores_for_indices(labels, final_pred, canonical_test_idx),
                "delta": _semantic_gate_delta(labels, base_pred, final_pred, canonical_test_idx),
            },
            "candidate_oracle": {},
            "trainable_gate_params": int(
                sum(param.numel() for param in model.parameters() if param.requires_grad)
                + (sum(param.numel() for param in break_model.parameters() if param.requires_grad) if break_model is not None else 0)
            ),
            "cuda_max_memory_allocated": _max_cuda_memory_allocated(device),
        }
        for action_idx, name in enumerate(candidate_names):
            cand_final = base_pred.clone()
            cand_final[test_idx] = candidate_preds[action_idx][test_idx]
            metrics["candidate_oracle"][name] = {
                "test_candidate": _scores_for_indices(labels, candidate_preds[action_idx], test_idx),
                "test_delta_if_always_take": _semantic_gate_delta(labels, base_pred, cand_final, test_idx),
                "positive_utility_train": int((targets[train_idx, action_idx] > 0).sum().item()),
                "positive_utility_valid": int((targets[valid_idx, action_idx] > 0).sum().item()),
                "positive_utility_test": int((targets[test_idx, action_idx] > 0).sum().item()),
            }
        outputs = {
            "gate_logits": gate_logits,
            "gate_prob": gate_prob,
            "action_logits": action_logits,
            "action_prob": action_prob,
            "break_logits": break_logits,
            "break_prob": break_prob,
            "final_pred": final_pred,
            "base_pred": base_pred,
            "labels": labels,
            "selected_action": selected_action,
            "selected_score": selected_score,
            "candidate_names": list(candidate_names),
            "candidate_preds": torch.stack(candidate_preds, dim=1),
            "candidate_probs": torch.stack(candidate_probs, dim=1),
            "targets": targets,
            "defer_targets": defer_targets,
            "rewards": rewards,
            "node_attribute_features": node_attribute_features,
            "local_competence_features": local_competence_features,
        }
        write_torch(stage_dir / "outputs.pt", outputs)
        write_torch(
            stage_dir / "checkpoint.pt",
            {
                "model": best_state,
                "break_model": best_break_state,
                "model_params": {
                    "in_channels": int(features.shape[-1]),
                    "hidden_channels": int(getattr(args, "semantic_gate_hidden_dim", 64)),
                },
                "threshold": threshold,
                "break_threshold": break_threshold,
                "candidate_names": list(candidate_names),
                "feature_family": feature_family,
                "selection_policy": selection_policy,
                "safety_policy": safety_policy,
                "node_attribute_metadata": node_attribute_metadata,
                "local_competence_metadata": local_competence_metadata,
            },
        )
        write_json(stage_dir / "metrics.json", metrics)
        per_node_path = stage_dir / "per_node_test.jsonl"
        with per_node_path.open("w", encoding="utf-8") as handle:
            for node_idx in _as_long_cpu_tensor(test_idx).tolist():
                action_idx = int(selected_action[node_idx].item())
                row = {
                    "node_id": int(node_idx),
                    "label": int(labels[node_idx].item()),
                    "base_pred": int(base_pred[node_idx].item()),
                    "final_pred": int(final_pred[node_idx].item()),
                    "selected_action": action_idx,
                    "selected_action_name": "base" if action_idx < 0 else str(candidate_names[action_idx]),
                    "selected_score": float(selected_score[node_idx].item()),
                    "gate_probs": {
                        str(candidate_names[i]): float(gate_prob[node_idx, i].item())
                        for i in range(len(candidate_names))
                    },
                    "action_probs": (
                        {
                            "base": float(action_prob[node_idx, 0].item()),
                            **{
                                str(candidate_names[i]): float(action_prob[node_idx, i + 1].item())
                                for i in range(len(candidate_names))
                            },
                        }
                        if action_prob is not None
                        else None
                    ),
                    "break_probs": (
                        {
                            str(candidate_names[i]): float(break_prob[node_idx, i].item())
                            for i in range(len(candidate_names))
                        }
                        if break_prob is not None
                        else None
                    ),
                    "base_correct": bool(base_pred[node_idx].item() == labels[node_idx].item()),
                    "final_correct": bool(final_pred[node_idx].item() == labels[node_idx].item()),
                    "changed": bool(base_pred[node_idx].item() != final_pred[node_idx].item()),
                    "fix": bool(base_pred[node_idx].item() != labels[node_idx].item() and final_pred[node_idx].item() == labels[node_idx].item()),
                    "break": bool(base_pred[node_idx].item() == labels[node_idx].item() and final_pred[node_idx].item() != labels[node_idx].item()),
                }
                if local_competence_features is not None and local_competence_metadata is not None:
                    local_names = list(local_competence_metadata.get("feature_names", []))
                    row["local_competence"] = {
                        str(candidate_names[action_i]): {
                            local_names[feature_i]: float(local_competence_features[node_idx, action_i, feature_i].item())
                            for feature_i in range(len(local_names))
                        }
                        for action_i in range(len(candidate_names))
                    }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest.update(
            {
                "status": "completed",
                "routed_nodes": routed_info,
                "feature_dim": int(features.shape[-1]),
                "candidate_count": int(len(candidate_names)),
                "epochs": int(epochs),
                "validation_threshold": threshold_row,
                "feature_family": feature_family,
                "selection_policy": selection_policy,
                "safety_policy": safety_policy,
                "break_budget": float(getattr(args, "semantic_gate_break_budget", -1.0)),
                "node_attribute_metadata": node_attribute_metadata,
                "local_competence_metadata": local_competence_metadata,
                "outputs_path": str(stage_dir / "outputs.pt"),
                "checkpoint_path": str(stage_dir / "checkpoint.pt"),
                "metrics_path": str(stage_dir / "metrics.json"),
                "per_node_test_path": str(per_node_path),
            }
        )
        write_json(stage_dir / "manifest.json", manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"stage": "semantic_correction_gate", "stage_dir": str(stage_dir), "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(stage_dir / "manifest.json", manifest)
        raise


def run_semantic_embedding_classifier_seed(args, seed, data, experiment_root, run):
    stage_dir = build_preparation_dir(experiment_root, "semantic_embedding_classifier")
    code_provenance = capture_code_metadata(Path(__file__).resolve().parents[1])
    manifest = {
        "contract": "semantic_embedding_classifier_v1",
        "status": "started",
        "seed": int(seed),
        "canonical_task_name": "semantic_embedding_classifier",
        "dataset": getattr(args, "dataset", "unknown"),
        "dataset_path": str(data.get("dataset_path", "")),
        "training_scope": "train_idx_supervised_cached_embedding_classifier",
        "notes": "Direct classification over cached semantic embeddings under the frozen SimTeG/LMBot contract.",
        "command": _resolve_embedding_classifier_command(args),
        "code_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
        "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
        "stage_visibility": "public",
        "artifact_namespace": "preparation/semantic_embedding_classifier",
        "invocation": {
            "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
            "resolved_task": "semantic_embedding_classifier",
        },
    }
    write_json(stage_dir / "manifest.json", manifest)
    write_text(stage_dir / "command.txt", manifest["command"] + "\n")

    try:
        device = _resolve_device(getattr(args, "device", -1))
        if device.type == "cuda":
            _reset_cuda_peak_memory_stats(device)
        labels = _labels_to_index(data["labels"]).cpu()
        embeddings, embedding_path, embedding_source = _resolve_cached_semantic_embeddings(args, data, seed)
        if int(embeddings.shape[0]) != int(labels.numel()):
            raise ValueError(
                f"semantic_embedding_classifier row mismatch: embeddings={int(embeddings.shape[0])}, labels={int(labels.numel())}"
            )
        train_idx, valid_idx, test_idx, routed_info, training_scope = _resolve_semantic_split_indices(args, data, seed)
        _, semantic_text_source = _resolve_semantic_texts(args, data)
        manifest["training_scope"] = (
            "routed_nodes_supervised_cached_embedding_classifier" if routed_info is not None else "train_idx_supervised_cached_embedding_classifier"
        )
        manifest["semantic_text_source"] = semantic_text_source
        if routed_info is not None:
            manifest["routed_nodes"] = routed_info
        if train_idx.numel() == 0:
            raise ValueError("semantic_embedding_classifier requires a non-empty train_idx.")

        model = MLP(
            in_channels=int(embeddings.shape[1]),
            hidden_channels=int(getattr(args, "LM_classifier_hidden_dim", 128)),
            out_channels=2,
            num_layers=int(getattr(args, "LM_classifier_n_layers", 2)),
            dropout=float(getattr(args, "dropout", 0.4)),
            act=str(getattr(args, "activation", "leakyrelu")),
            norm="batch_norm",
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(getattr(args, "lr_LM", getattr(args, "lm_learning_rate", 1e-5))),
            weight_decay=float(getattr(args, "weight_decay_LM", getattr(args, "lm_weight_decay", 0.01))),
        )
        max_epochs = max(int(getattr(args, "LM_pretrain_epochs", 5)), 1)
        losses = []
        best_state = None
        best_valid = (-float("inf"), -float("inf"))
        best_epoch = -1
        x_all = embeddings.to(device)
        y_all = labels.to(device)
        train_x = x_all[train_idx.to(device)]
        train_y = y_all[train_idx.to(device)]
        valid_idx_device = valid_idx.to(device)

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits_train = model(train_x)
            loss = F.cross_entropy(logits_train, train_y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            run.log({"semantic_embedding_classifier_loss": loss_value, "semantic_embedding_classifier_epoch": epoch + 1})

            model.eval()
            with torch.no_grad():
                logits_all = model(x_all).cpu()
            valid_scores = _classification_metrics_from_logits(logits_all, labels, valid_idx)
            score_tuple = (float(valid_scores["macro_f1"]), float(valid_scores["accuracy"]))
            if score_tuple > best_valid:
                best_valid = score_tuple
                best_epoch = int(epoch)
                best_state = {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "valid_scores": dict(valid_scores),
                }

        if best_state is None:
            raise RuntimeError("semantic_embedding_classifier failed to record a best checkpoint.")

        model.load_state_dict(best_state["model"])
        model.eval()
        with torch.no_grad():
            logits = model(x_all).cpu()
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        outputs = {"logits": logits, "prob": prob, "pred": pred, "labels": labels}

        write_torch(stage_dir / "outputs.pt", outputs)
        write_torch(
            stage_dir / "classifier.pt",
            {
                "model": best_state["model"],
                "model_params": {
                    "in_channels": int(embeddings.shape[1]),
                    "hidden_channels": int(getattr(args, "LM_classifier_hidden_dim", 128)),
                    "out_channels": 2,
                    "num_layers": int(getattr(args, "LM_classifier_n_layers", 2)),
                    "dropout": float(getattr(args, "dropout", 0.4)),
                    "act": str(getattr(args, "activation", "leakyrelu")),
                    "norm": "batch_norm",
                },
            },
        )
        write_torch(stage_dir / "embeddings_ref.pt", embeddings)

        metrics = {
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "best_epoch": int(best_epoch),
            "train": _classification_metrics_from_logits(logits, labels, train_idx),
            "validation": _classification_metrics_from_logits(logits, labels, valid_idx),
            "test": _classification_metrics_from_logits(logits, labels, test_idx),
            "trainable_classifier_params": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            "cuda_max_memory_allocated": _max_cuda_memory_allocated(device),
        }
        write_json(stage_dir / "metrics.json", metrics)
        manifest.update(
            {
                "status": "completed",
                "embedding_path": str(embedding_path),
                "embedding_source": str(embedding_source),
                "semantic_text_source": semantic_text_source,
                "embedding_dim": int(embeddings.shape[1]),
                "embedding_row_count": int(embeddings.shape[0]),
                "train_limit": int(train_idx.numel()),
                "epochs": int(max_epochs),
                "classifier_path": str(stage_dir / "classifier.pt"),
                "outputs_path": str(stage_dir / "outputs.pt"),
                "metrics_path": str(stage_dir / "metrics.json"),
                "architecture": {
                    "type": "cached_embedding_mlp",
                    "hidden_dim": int(getattr(args, "LM_classifier_hidden_dim", 128)),
                    "num_layers": int(getattr(args, "LM_classifier_n_layers", 2)),
                    "dropout": float(getattr(args, "dropout", 0.4)),
                    "activation": str(getattr(args, "activation", "leakyrelu")),
                },
            }
        )
        write_json(stage_dir / "manifest.json", manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"stage": "semantic_embedding_classifier", "stage_dir": str(stage_dir), "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(stage_dir / "manifest.json", manifest)
        raise


def _select_semantic_train_idx(train_idx, limit, seed):
    train_idx = _as_long_cpu_tensor(train_idx)
    if int(limit or 0) <= 0 or int(limit) >= int(train_idx.numel()):
        return train_idx
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(train_idx.numel(), generator=generator)
    return train_idx[order[: int(limit)]]


def run_semantic_finetune_seed(args, seed, data, experiment_root, run):
    stage_dir = build_preparation_dir(experiment_root, "semantic_encoder")
    code_provenance = capture_code_metadata(Path(__file__).resolve().parents[1])
    manifest = {
        "contract": "semantic_finetune_v2",
        "status": "started",
        "seed": int(seed),
        "canonical_task_name": "semantic_encoder_finetune",
        "semantic_backbone": getattr(args, "semantic_encoder", getattr(args, "semantic_backbone", "auto")),
        "dataset": getattr(args, "dataset", "unknown"),
        "dataset_path": str(data.get("dataset_path", "")),
        "training_scope": "train_idx_supervised_semantic_backbone",
        "notes": "Exploratory semantic finetune unless promoted by later full validation.",
        "command": _semantic_command(args),
        "code_commit": code_provenance["commit"],
        "code_provenance": code_provenance,
        "deprecated_cli_flags": list(getattr(args, "deprecated_cli_flags", [])),
        "stage_visibility": "public",
        "artifact_namespace": "preparation/semantic_encoder",
        "invocation": {
            "requested_task": getattr(args, "requested_experiment_task", getattr(args, "experiment_task", None)),
            "resolved_task": "semantic_encoder_finetune",
        },
    }
    write_json(stage_dir / "manifest.json", manifest)
    write_text(stage_dir / "command.txt", manifest["command"] + "\n")

    try:
        device = _resolve_device(getattr(args, "device", -1))
        if device.type == "cuda":
            _reset_cuda_peak_memory_stats(device)
        labels = _labels_to_index(data["labels"]).cpu()
        user_text, semantic_text_source = _resolve_semantic_texts(args, data)
        train_idx, valid_idx, test_idx, routed_info, training_scope = _resolve_semantic_split_indices(args, data, seed)
        manifest["training_scope"] = training_scope
        manifest["semantic_text_source"] = semantic_text_source
        if routed_info is not None:
            manifest["routed_nodes"] = routed_info
        if train_idx.numel() == 0:
            raise ValueError("semantic_finetune requires a non-empty train_idx.")

        lm_model = _semantic_backbone_to_lm_model(args)
        supervision_mode = _resolve_semantic_supervision_mode(args, lm_model)
        model_config = {
            "lm_model": lm_model,
            "qwen_model_path": getattr(args, "qwen_model_path", None),
            "dropout": getattr(args, "dropout", 0.4),
            "att_dropout": getattr(args, "LM_att_dropout", 0.1),
            "lm_dropout": getattr(args, "LM_dropout", 0.1),
            "classifier_n_layers": getattr(args, "LM_classifier_n_layers", 2),
            "classifier_hidden_dim": getattr(args, "LM_classifier_hidden_dim", 128),
            "activation": getattr(args, "activation", "leakyrelu"),
            "device": device,
            "peft_rank": getattr(args, "peft_rank", 8),
            "peft_alpha": getattr(args, "peft_alpha", 16.0),
            "qwen_trust_remote_code": getattr(args, "qwen_trust_remote_code", False),
            "detach_embeddings": False,
        }
        model, tokenizer = build_LM_model(model_config)
        answer_token_ids = None
        checkpoint_summary = None
        if lm_model == "roberta_finetuned":
            checkpoint_path = _resolve_finetuned_roberta_checkpoint(args, seed)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "semantic_encoder_finetune with --semantic_encoder roberta_finetuned requires a SimTeG LM checkpoint. "
                    "Pass --finetuned_roberta_checkpoint_path or ensure TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl exists."
                )
            checkpoint_summary = _load_simteg_lm_checkpoint_into_encoder(model.LM, checkpoint_path)
            checkpoint_summary["checkpoint_path"] = str(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if supervision_mode == "answer_token":
            answer_token_ids = _semantic_answer_token_sequences(tokenizer)
            for param in model.classifier.parameters():
                param.requires_grad = False
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(getattr(args, "lr_LM", 1e-5)),
            weight_decay=float(getattr(args, "weight_decay_LM", 0.01)),
        )
        batch_size = max(int(getattr(args, "batch_size_LM", 1)), 1)
        max_length = int(getattr(args, "max_length", 512))
        max_steps = int(getattr(args, "semantic_max_steps", 0) or np.ceil(train_idx.numel() / batch_size))
        max_steps = max(max_steps, 1)
        losses = []
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = train_idx[torch.randperm(train_idx.numel(), generator=generator)]

        for step in range(max_steps):
            start = (step * batch_size) % int(order.numel())
            if start == 0 and step > 0:
                order = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
            batch_idx = order[start : start + batch_size]
            if batch_idx.numel() < batch_size:
                extra = order[: batch_size - batch_idx.numel()]
                batch_idx = torch.cat([batch_idx, extra])
            texts = [user_text[int(idx)] for idx in batch_idx]
            y = labels[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            if supervision_mode == "answer_token":
                train_batch = _build_answer_token_train_batch(
                    tokenizer,
                    texts,
                    y.detach().cpu().tolist(),
                    max_length=max_length,
                    device=device,
                    answer_token_ids=answer_token_ids,
                )
                out = model.LM(**train_batch, use_cache=False)
                loss = out.loss
            else:
                tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
                _, logits = model(tokenized)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            run.log({"semantic_finetune_loss": loss_value, "semantic_finetune_step": step + 1})

        model.eval()
        eval_idx = _semantic_eval_indices(train_idx, valid_idx, test_idx, len(user_text))
        all_embeddings = []
        all_logits = []
        with torch.no_grad():
            for start in range(0, int(eval_idx.numel()), batch_size):
                batch_eval_idx = eval_idx[start : start + batch_size]
                texts = [user_text[int(idx)] for idx in batch_eval_idx]
                if supervision_mode == "answer_token":
                    batch_embeddings, batch_logits = _score_answer_token_candidates(
                        model,
                        tokenizer,
                        texts,
                        max_length=max_length,
                        device=device,
                        answer_token_ids=answer_token_ids,
                    )
                else:
                    tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
                    batch_embeddings, batch_logits = model(tokenized)
                all_embeddings.append(batch_embeddings.cpu())
                all_logits.append(batch_logits.cpu())
        eval_embeddings = torch.cat(all_embeddings, dim=0)
        eval_logits = torch.cat(all_logits, dim=0)
        embeddings = torch.zeros(
            (len(user_text), int(eval_embeddings.shape[1])),
            dtype=eval_embeddings.dtype,
        )
        logits = torch.zeros(
            (len(user_text), int(eval_logits.shape[1])),
            dtype=eval_logits.dtype,
        )
        embeddings[eval_idx] = eval_embeddings
        logits[eval_idx] = eval_logits
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        outputs = {
            "logits": logits,
            "prob": prob,
            "pred": pred,
            "labels": labels,
            "evaluated_node_ids": eval_idx,
            "semantic_supervision_mode": supervision_mode,
        }

        write_torch(stage_dir / "embeddings.pt", embeddings)
        write_torch(stage_dir / "outputs.pt", outputs)
        write_torch(stage_dir / "classifier.pt", model.classifier.state_dict())
        if lm_model == "qwen3_peft":
            adapter_dir = stage_dir / "adapter"
            model.LM.save_pretrained(adapter_dir)
            model_artifact = {"adapter_dir": str(adapter_dir)}
        else:
            write_torch(stage_dir / "model.pt", {"model": model.state_dict(), "model_config": model_config})
            model_artifact = {"model_path": str(stage_dir / "model.pt")}

        metrics = {
            "losses": losses,
            "final_loss": losses[-1],
            "train": _classification_metrics_from_logits(logits, labels, train_idx),
            "validation": _classification_metrics_from_logits(logits, labels, valid_idx),
            "test": _classification_metrics_from_logits(logits, labels, test_idx),
            "trainable_model_params": int(sum(param.numel() for param in model.LM.parameters() if param.requires_grad)),
            "trainable_classifier_params": int(sum(param.numel() for param in model.classifier.parameters() if param.requires_grad)),
            "semantic_supervision_mode": supervision_mode,
            "cuda_max_memory_allocated": _max_cuda_memory_allocated(device),
        }
        write_json(stage_dir / "metrics.json", metrics)
        manifest.update(
            {
                "status": "completed",
                "lm_model": lm_model,
                "semantic_supervision_mode": supervision_mode,
                "prompt_answer_schema": "ASSISTANT_ANSWER: Yes|No",
                "prompt_answer_slot": _SEMANTIC_ANSWER_SLOT,
                "answer_prompt_contract": "prompt_must_end_at_final_assistant_answer_slot",
                "answer_prompt_tokenization": "prompt_special_tokens_disabled_manual_yes_no_suffix",
                "answer_token_mapping": _semantic_answer_token_mapping_manifest() if supervision_mode == "answer_token" else None,
                "qwen_model_path": getattr(args, "qwen_model_path", None),
                "qwen_trust_remote_code": bool(getattr(args, "qwen_trust_remote_code", False)),
                "semantic_text_source": semantic_text_source,
                "train_limit": int(train_idx.numel()),
                "eval_node_count": int(eval_idx.numel()),
                "eval_row_layout": "full_graph_zero_fill_for_non_eval_nodes",
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "embeddings_path": str(stage_dir / "embeddings.pt"),
                "outputs_path": str(stage_dir / "outputs.pt"),
                "classifier_path": str(stage_dir / "classifier.pt"),
                "classifier_role": "trainable_head" if supervision_mode == "classifier" else "compatibility_artifact_unused_by_answer_token",
                "metrics_path": str(stage_dir / "metrics.json"),
                "finetuned_roberta_checkpoint_path": str((checkpoint_summary or {}).get("checkpoint_path", "")),
                "finetuned_roberta_checkpoint_load": checkpoint_summary,
                **model_artifact,
            }
        )
        write_json(stage_dir / "manifest.json", manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"stage": "semantic_encoder_finetune", "stage_dir": str(stage_dir), "metrics": metrics}
    except Exception as exc:
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json(stage_dir / "manifest.json", manifest)
        raise
