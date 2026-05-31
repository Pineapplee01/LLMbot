from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn.models import MLP

from model_building import _labels_to_index, build_LM_model
from runtime_env import _max_cuda_memory_allocated, _reset_cuda_peak_memory_stats, _resolve_device
from utils import build_preparation_dir, capture_code_metadata, safe_torch_load, write_json, write_text, write_torch


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


def _semantic_command(args):
    keys = [
        "experiment_task",
        "dataset",
        "reset_split",
        "seeds",
        "semantic_encoder",
        "qwen_model_path",
        "qwen_trust_remote_code",
        "peft_rank",
        "peft_alpha",
        "lm_batch_size",
        "max_length",
        "semantic_train_limit",
        "semantic_max_steps",
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


def _semantic_tokenize(tokenizer, texts, max_length, device):
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    return {key: value.to(device) for key, value in tokenized.items()}


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


def _resolve_embedding_classifier_command(args):
    keys = [
        "experiment_task",
        "dataset",
        "reset_split",
        "seeds",
        "embedding_path",
        "semantic_encoder",
        "LM_classifier_n_layers",
        "LM_classifier_hidden_dim",
        "dropout",
        "lm_learning_rate",
        "lm_weight_decay",
        "semantic_train_limit",
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
        train_idx = _select_semantic_train_idx(
            data["train_idx"],
            getattr(args, "semantic_train_limit", 0),
            seed,
        )
        valid_idx = _as_long_cpu_tensor(data["valid_idx"])
        test_idx = _as_long_cpu_tensor(data["test_idx"])
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
        "contract": "semantic_finetune_v1",
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
        user_text = list(data["user_text"])
        train_idx = _select_semantic_train_idx(
            data["train_idx"],
            getattr(args, "semantic_train_limit", 0),
            seed,
        )
        valid_idx = _as_long_cpu_tensor(data["valid_idx"])
        test_idx = _as_long_cpu_tensor(data["test_idx"])
        if train_idx.numel() == 0:
            raise ValueError("semantic_finetune requires a non-empty train_idx.")

        lm_model = _semantic_backbone_to_lm_model(args)
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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
            tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(tokenized)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            run.log({"semantic_finetune_loss": loss_value, "semantic_finetune_step": step + 1})

        model.eval()
        all_embeddings = []
        all_logits = []
        with torch.no_grad():
            for start in range(0, len(user_text), batch_size):
                texts = user_text[start : start + batch_size]
                tokenized = _semantic_tokenize(tokenizer, texts, max_length, device)
                embeddings, logits = model(tokenized)
                all_embeddings.append(embeddings.cpu())
                all_logits.append(logits.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)
        logits = torch.cat(all_logits, dim=0)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        outputs = {"logits": logits, "prob": prob, "pred": pred, "labels": labels}

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
            "cuda_max_memory_allocated": _max_cuda_memory_allocated(device),
        }
        write_json(stage_dir / "metrics.json", metrics)
        manifest.update(
            {
                "status": "completed",
                "lm_model": lm_model,
                "qwen_model_path": getattr(args, "qwen_model_path", None),
                "qwen_trust_remote_code": bool(getattr(args, "qwen_trust_remote_code", False)),
                "train_limit": int(train_idx.numel()),
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "embeddings_path": str(stage_dir / "embeddings.pt"),
                "outputs_path": str(stage_dir / "outputs.pt"),
                "classifier_path": str(stage_dir / "classifier.pt"),
                "metrics_path": str(stage_dir / "metrics.json"),
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
