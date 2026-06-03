"""Precompute prompt caches for GLANCE-style, relation-aware, and expert views.

This helper builds per-node prompt texts, optionally generates explanation-first
expert summaries, encodes them with a local embedding model, applies
encoder-aligned pooling plus optional l2 normalization, and stores a prompt
cache under a stable ``embeddings`` key for downstream graph_detector_prepare
and strict GLANCE runs.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import ensure_dir, read_json, resolve_dataset_path, tensor_sha256, write_json, write_torch


DEFAULT_QWEN_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/"
    "snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
)
DEFAULT_FINETUNED_ROBERTA_MODEL_PATH = "yzxjb/roberta-finetuned-20"
DEFAULT_FINETUNED_ROBERTA_MODEL_ALIAS = "roberta_finetuned"
DEFAULT_OUTPUT_NAME = "glance_qwen3_prompt_cache.pt"
DEFAULT_ACCOUNT_REFERENCE_DATE = datetime(2020, 9, 1)
LEGACY_PROMPT_MODES = (
    "glance_ego",
    "glance_hop1",
    "glance_hop2",
    "glance_concat_ego_hop1_hop2",
    "relation_aware_ego",
    "relation_aware_1hop",
)
EXPERT_PROMPT_MODES = (
    "expert_ego",
    "expert_graph_following",
    "expert_graph_follower",
    "expert_tweet",
    "expert_conflict",
    "expert_concat_v1",
)
PROMPT_MODE_CHOICES = LEGACY_PROMPT_MODES + EXPERT_PROMPT_MODES
PROMPT_FAMILY_VERSION_CHOICES = ("v1", "v2")
NEIGHBOR_SAMPLING_CHOICES = ("auto", "uniform_random", "directional_heuristic", "center_induced_relation_aware")
CENTER_NODE_SCOPE_CHOICES = ("labeled", "all_graph_nodes")
EXPERT_COMPONENT_NAMES = ("ego", "graph_following", "graph_follower", "tweet", "conflict")
EXPERT_STRUCTURED_COMPONENT_NAMES = ("metadata_structured",)
EXPERT_COMPONENT_NAMES_WITH_STRUCTURED = EXPERT_COMPONENT_NAMES + EXPERT_STRUCTURED_COMPONENT_NAMES
EXPERT_COMPONENT_NAMES_V2 = ("graph_following", "graph_follower", "tweet", "conflict")
EXPERT_COMPONENT_NAMES_V2_WITH_STRUCTURED = EXPERT_COMPONENT_NAMES_V2 + EXPERT_STRUCTURED_COMPONENT_NAMES
EXPERT_SCALAR_KEYS = (
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
)
GRAPH_DATA_VARIANT_CHOICES = ("labeled", "full_graph_support")
TWEET_SOURCE_MODE_CHOICES = ("norm_user_text", "raw_post_edges")
TWEET_CLEAN_LEVEL_CHOICES = ("light", "norm_compatible")
_MODEL_SOURCE_ALIASES = {
    "finetuned_roberta": DEFAULT_FINETUNED_ROBERTA_MODEL_PATH,
    "roberta_finetuned": DEFAULT_FINETUNED_ROBERTA_MODEL_PATH,
    "roberta-f": DEFAULT_FINETUNED_ROBERTA_MODEL_PATH,
}
_ACTIVE_WANDB_RUN = None
_ACTIVE_WANDB_FINISHED = False


class _NullWandbRun:
    def log(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None


def _wandb_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_wandb_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _wandb_config_value(item) for key, item in value.items()}
    return str(value)


def _setup_precompute_wandb(args):
    global _ACTIVE_WANDB_RUN, _ACTIVE_WANDB_FINISHED
    if bool(getattr(args, "disable_wandb", False)):
        return _NullWandbRun()
    project_name = str(getattr(args, "project_name", "") or "").strip()
    if not project_name:
        return _NullWandbRun()
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "precompute.py wandb monitoring was requested, but wandb is not installed. "
            "Install wandb in the active environment or pass --disable_wandb."
        ) from exc
    run_name = str(getattr(args, "wandb_run_name", "") or "").strip()
    if not run_name:
        run_name = (
            f"{getattr(args, 'experiment_name', 'precompute')}_"
            f"{getattr(args, 'prompt_mode', 'prompt')}_seed_{int(getattr(args, 'seed', 0))}"
        )
    config = {
        key: _wandb_config_value(value)
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        job_type="prompt_precompute",
    )
    _ACTIVE_WANDB_RUN = run
    _ACTIVE_WANDB_FINISHED = False
    return run


def _finish_precompute_wandb(run=None, exit_code=0):
    global _ACTIVE_WANDB_RUN, _ACTIVE_WANDB_FINISHED
    run = run if run is not None else _ACTIVE_WANDB_RUN
    if run is None or isinstance(run, _NullWandbRun) or _ACTIVE_WANDB_FINISHED:
        return
    finish_fn = getattr(run, "finish", None)
    if finish_fn is None:
        return
    try:
        finish_fn(exit_code=int(exit_code))
    finally:
        _ACTIVE_WANDB_FINISHED = True
        if run is _ACTIVE_WANDB_RUN:
            _ACTIVE_WANDB_RUN = None


atexit.register(_finish_precompute_wandb)


def _wandb_log(run, payload, step=None):
    if run is None:
        return
    log_fn = getattr(run, "log", None)
    if log_fn is None:
        return
    try:
        if step is None:
            log_fn(payload)
        else:
            log_fn(payload, step=int(step))
    except Exception as exc:
        print({"wandb_warning": str(exc)})


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum().item() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def decoder_generation_continuation(output_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Mirror Qwen's official chat-generation slicing: output_ids[len(input_ids):]."""
    prompt_length = int(input_ids.shape[-1])
    if int(output_ids.shape[0]) <= prompt_length:
        return output_ids[0:0]
    return output_ids[prompt_length:]


def masked_mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(dtype=last_hidden_states.dtype)
    summed = (last_hidden_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).replace("</s>", " ").strip().lower()
    if " " in normalized:
        normalized = normalized.split()[0]
    if normalized in {"", "none", "null", "nan"}:
        return False
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _safe_bool_field(value, default=False):
    try:
        return bool(_parse_bool(value))
    except Exception:
        return bool(default)


def _load_texts(dataset_path: Path, text_path: Path | None):
    path = Path(text_path) if text_path else dataset_path / "norm_user_text.json"
    texts = read_json(path, default=None)
    if texts is None:
        raise FileNotFoundError(f"Could not read text cache: {path}")
    if not isinstance(texts, list):
        raise ValueError(f"Expected a list of node texts in {path}, got {type(texts).__name__}.")
    return texts, path


def _iter_node_new_object(path: Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = ""
        cursor = 0
        started = False
        eof = False
        while True:
            if cursor:
                buffer = buffer[cursor:]
                cursor = 0
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not started:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if eof:
                        return
                    continue
                if buffer[cursor] != "{":
                    raise ValueError(f"Expected a top-level JSON object in {path}")
                cursor += 1
                started = True
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    break
                current = buffer[cursor]
                if current == ",":
                    cursor += 1
                    continue
                if current == "}":
                    return
                try:
                    key, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                cursor = end
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer) or buffer[cursor] != ":":
                    break
                cursor += 1
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                cursor = end
                yield key, value
            if eof:
                return


def _iter_json_array(path: Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = ""
        cursor = 0
        started = False
        eof = False
        while True:
            if cursor:
                buffer = buffer[cursor:]
                cursor = 0
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not started:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    if eof:
                        return
                    continue
                if buffer[cursor] != "[":
                    raise ValueError(f"Expected a top-level JSON array in {path}")
                cursor += 1
                started = True
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    break
                current = buffer[cursor]
                if current == ",":
                    cursor += 1
                    continue
                if current == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                cursor = end
                yield value
            if eof:
                return


def _load_feature_tensor(path: Path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("embeddings", "features", "x"):
            if key in payload:
                payload = payload[key]
                break
    if not torch.is_tensor(payload):
        payload = torch.as_tensor(payload)
    payload = payload.detach().cpu().float()
    if payload.dim() != 2:
        raise ValueError(f"Expected a 2-D embedding tensor at {path}, got shape {tuple(payload.shape)}.")
    return payload.contiguous()


def _resolve_default_selection_embedding_path(dataset_path: Path, seed: int):
    candidates = [
        dataset_path / f"embeddings_iter_-1_seed_{int(seed)}.pt",
        dataset_path / "embeddings_roberta.pt",
        dataset_path / "finetuned_roberta_embeddings_iter_2_seed1.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _canonical_node_id(raw_id):
    text = str(raw_id).strip()
    if not text:
        return ""
    if text.startswith("u") or text.startswith("t"):
        return text
    return f"u{text}"


def _resolve_default_node_source_path(dataset_path: Path):
    return dataset_path / "node_new.json"


def _resolve_default_edge_source_path(dataset_path: Path):
    edge_new = dataset_path / "edge_new.json"
    if edge_new.exists():
        return edge_new
    return dataset_path / "edge.csv"


def _build_index_to_user_id(dataset_path: Path, graph_data_variant: str, center_node_scope: str, target_node_ids):
    target_node_ids = list(int(item) for item in target_node_ids)
    split = read_json(dataset_path / "split_new.json", default={}) or {}
    train_ids = list(split.get("train") or [])
    valid_ids = list(split.get("val") or split.get("valid") or split.get("dev") or [])
    test_ids = list(split.get("test") or [])
    labeled_ids = [_canonical_node_id(item) for item in (train_ids + valid_ids + test_ids)]
    if graph_data_variant == "labeled" or (target_node_ids and max(target_node_ids) < len(labeled_ids)):
        return {int(idx): labeled_ids[int(idx)] for idx in target_node_ids}
    support_path = dataset_path / "support.json"
    support_ids = []
    for record in _iter_json_array(support_path, 1 << 20):
        raw_id = record.get("ID") or record.get("id") or record.get("user_id")
        support_ids.append(_canonical_node_id(raw_id))
    all_ids = labeled_ids + support_ids
    return {int(idx): all_ids[int(idx)] for idx in target_node_ids}


def _build_raw_node_indices(node_source_path: Path, target_user_ids=None, target_tweet_ids=None, need_tweets=True):
    target_user_ids = None if target_user_ids is None else {str(item) for item in target_user_ids}
    target_tweet_ids = None if target_tweet_ids is None else {str(item) for item in target_tweet_ids}
    user_profile_by_id = {}
    tweet_text_by_id = {}
    for node_id, payload in _iter_node_new_object(node_source_path):
        node_id = str(node_id)
        if node_id.startswith("u"):
            if target_user_ids is not None and node_id not in target_user_ids:
                continue
            if not isinstance(payload, dict):
                continue
            user_profile_by_id[node_id] = {
                "created_at": _compact_whitespace(payload.get("created_at")),
                "description": str(payload.get("description") or ""),
                "location": _compact_whitespace(payload.get("location")),
                "name": _compact_whitespace(payload.get("name")),
                "username": _compact_whitespace(payload.get("username")),
                "verified": _safe_bool_field(payload.get("verified"), default=False),
                "protected": _safe_bool_field(payload.get("protected"), default=False),
                "public_metrics": dict(payload.get("public_metrics") or {}),
            }
        elif need_tweets and node_id.startswith("t"):
            if target_tweet_ids is not None and node_id not in target_tweet_ids:
                continue
            if not isinstance(payload, dict):
                continue
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                tweet_text_by_id[node_id] = text
        if target_user_ids is not None and len(user_profile_by_id) >= len(target_user_ids):
            users_done = True
        else:
            users_done = False
        if target_tweet_ids is not None and len(tweet_text_by_id) >= len(target_tweet_ids):
            tweets_done = True
        else:
            tweets_done = False
        if (target_user_ids is not None or target_tweet_ids is not None) and ((target_user_ids is None or users_done) and (target_tweet_ids is None or tweets_done)):
            break
    return {
        "user_profile_by_id": user_profile_by_id,
        "tweet_text_by_id": tweet_text_by_id,
    }


def _resolve_selection_feature_bundle(args, dataset_path: Path, graph_variant: str, graph_node_count: int, labeled_node_count: int):
    explicit_labeled = getattr(args, "selection_embedding_path", None)
    labeled_path = Path(explicit_labeled) if explicit_labeled else _resolve_default_selection_embedding_path(dataset_path, int(args.seed))
    if labeled_path is None or not labeled_path.exists():
        raise FileNotFoundError(
            "center_induced_relation_aware requires --selection_embedding_path or a default labeled embedding tensor "
            f"(for example embeddings_iter_-1_seed_{int(args.seed)}.pt) under {dataset_path}."
        )
    labeled_features = _load_feature_tensor(labeled_path)

    if graph_variant == "full_graph_support":
        if int(labeled_features.shape[0]) == int(graph_node_count):
            return {
                "features": F.normalize(labeled_features, p=2, dim=1, eps=1e-12),
                "mode": "single_full_graph_tensor",
                "labeled_path": str(labeled_path),
                "support_path": "",
            }
        if int(labeled_features.shape[0]) != int(labeled_node_count):
            raise ValueError(
                "full_graph_support center-induced selection expects labeled selection embeddings to have either "
                f"{labeled_node_count} rows or {graph_node_count} rows, got {int(labeled_features.shape[0])}."
            )
        explicit_support = getattr(args, "support_selection_embedding_path", None)
        support_path = Path(explicit_support) if explicit_support else dataset_path / "support_roberta_embeddings_new.pt"
        if not support_path.exists():
            raise FileNotFoundError(
                "full_graph_support center-induced selection requires --support_selection_embedding_path or the "
                f"default support embedding tensor at {support_path}."
            )
        support_features = _load_feature_tensor(support_path)
        expected_support = int(graph_node_count) - int(labeled_node_count)
        if int(support_features.shape[0]) != expected_support:
            raise ValueError(
                "Support selection embeddings do not match the full-graph support suffix size: "
                f"expected {expected_support}, got {int(support_features.shape[0])}."
            )
        if int(support_features.shape[1]) != int(labeled_features.shape[1]):
            raise ValueError(
                "Labeled and support selection embeddings must share the same feature dimension for "
                "center_induced_relation_aware."
            )
        full_features = torch.cat([labeled_features, support_features], dim=0)
        return {
            "features": F.normalize(full_features, p=2, dim=1, eps=1e-12),
            "mode": "runtime_labeled_plus_support_concat",
            "labeled_path": str(labeled_path),
            "support_path": str(support_path),
        }

    if int(labeled_features.shape[0]) != int(graph_node_count):
        raise ValueError(
            f"labeled graph variant expects selection embeddings with {graph_node_count} rows, "
            f"got {int(labeled_features.shape[0])}."
        )
    return {
        "features": F.normalize(labeled_features, p=2, dim=1, eps=1e-12),
        "mode": "single_labeled_tensor",
        "labeled_path": str(labeled_path),
        "support_path": "",
    }


def _resolve_graph_variant_paths(dataset_path: Path, graph_data_variant: str, text_path: Path | None):
    variant = str(graph_data_variant or "labeled").lower()
    if variant not in GRAPH_DATA_VARIANT_CHOICES:
        raise ValueError(f"Unsupported graph_data_variant: {graph_data_variant}")
    if variant == "full_graph_support":
        resolved_text_path = Path(text_path) if text_path else dataset_path / "norm_user_text_new.json"
        edge_index_path = dataset_path / "edge_index_new.pt"
        edge_type_path = dataset_path / "edge_type_new.pt"
    else:
        resolved_text_path = Path(text_path) if text_path else dataset_path / "norm_user_text.json"
        edge_index_path = dataset_path / "edge_index.pt"
        edge_type_path = dataset_path / "edge_type.pt"
    return {
        "variant": variant,
        "text_path": resolved_text_path,
        "edge_index_path": edge_index_path,
        "edge_type_path": edge_type_path,
    }


def _iter_post_edges(edge_source_path: Path):
    edge_source_path = Path(edge_source_path)
    if edge_source_path.suffix.lower() == ".json":
        for source_id, entries in _iter_node_new_object(edge_source_path):
            if str(source_id).lower() == "source_id":
                continue
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                relation = str(item[0]).strip().lower()
                target_id = str(item[1]).strip()
                if relation != "post":
                    continue
                yield str(source_id).strip(), relation, target_id
        return

    import csv

    with open(edge_source_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) >= 3 and str(row[0]).lower() in {"source_id", "source"}:
                continue
            if len(row) < 3:
                continue
            source_id = str(row[0]).strip()
            relation = str(row[1]).strip().lower()
            target_id = str(row[2]).strip()
            if relation != "post":
                continue
            yield source_id, relation, target_id


def _resolve_post_edge_direction(edge_source_path: Path, known_user_ids=None, known_tweet_ids=None, sample_limit: int = 20000):
    known_user_ids = None if known_user_ids is None else set(str(item) for item in known_user_ids)
    known_tweet_ids = None if known_tweet_ids is None else set(str(item) for item in known_tweet_ids)
    source_user_target_tweet = 0
    source_tweet_target_user = 0
    for idx, (source_id, _relation, target_id) in enumerate(_iter_post_edges(edge_source_path)):
        source_is_user = source_id.startswith("u") if known_user_ids is None else (source_id in known_user_ids)
        target_is_tweet = target_id.startswith("t") if known_tweet_ids is None else (target_id in known_tweet_ids)
        source_is_tweet = source_id.startswith("t") if known_tweet_ids is None else (source_id in known_tweet_ids)
        target_is_user = target_id.startswith("u") if known_user_ids is None else (target_id in known_user_ids)
        if source_is_user and target_is_tweet:
            source_user_target_tweet += 1
        if source_is_tweet and target_is_user:
            source_tweet_target_user += 1
        if idx + 1 >= int(sample_limit):
            break
    if source_user_target_tweet > source_tweet_target_user:
        return "source_to_target"
    if source_tweet_target_user > source_user_target_tweet:
        return "target_to_source"
    raise ValueError(
        "Could not resolve post-edge direction deterministically from edge.csv. "
        f"source_to_target={source_user_target_tweet}, target_to_source={source_tweet_target_user}."
    )


def _build_user_tweet_map(edge_source_path: Path, target_user_ids=None, target_tweet_ids=None, direction=None):
    if direction not in {"source_to_target", "target_to_source"}:
        raise ValueError("user-tweet map requires resolved post-edge direction.")
    target_user_ids = None if target_user_ids is None else {str(item) for item in target_user_ids}
    target_tweet_ids = None if target_tweet_ids is None else {str(item) for item in target_tweet_ids}
    user_tweet_map = {}
    for source_id, _relation, target_id in _iter_post_edges(edge_source_path):
        if direction == "source_to_target":
            user_id = source_id
            tweet_id = target_id
        else:
            user_id = target_id
            tweet_id = source_id
        if target_user_ids is not None and user_id not in target_user_ids:
            continue
        if target_tweet_ids is not None and tweet_id not in target_tweet_ids:
            continue
        user_tweet_map.setdefault(user_id, []).append(tweet_id)
    return user_tweet_map


def _hash_strings(values):
    digest = hashlib.sha256()
    digest.update(str(len(values)).encode("utf-8"))
    for item in values:
        digest.update(b"\0")
        digest.update(str(item).encode("utf-8"))
    return digest.hexdigest()


def _write_jsonl(path: Path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _string_sha256(text):
    digest = hashlib.sha256()
    digest.update(str(text or "").encode("utf-8"))
    return digest.hexdigest()


def _explanation_quality_issue(text):
    text = str(text or "").strip()
    if not text:
        return "empty"
    compact = "".join(text.split())
    if compact and compact.count("!") > max(30, int(len(compact) * 0.2)):
        return "many_exclamation_marks"
    lowered = text.lower()
    hard_echo_markers = (
        "write 4-6 evidence-grounded sentences",
        "do not output only a label",
        "system\nyou analyze twitter accounts",
    )
    if any(marker in lowered for marker in hard_echo_markers):
        return "prompt_echo"
    if re.search(r"(?im)^\s*(system|user|assistant)\s*:?\s*$", text):
        return "prompt_echo"
    if re.search(r"(?im)^\s*(system|user|assistant)\s*:\s+", text):
        return "prompt_echo"
    if len(text) < 24:
        return "too_short"
    return ""


def _is_explanation_quality_ok(text):
    return not _explanation_quality_issue(text)


def _coerce_node_id_sequence(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_coerce_node_id_sequence(item))
        return out
    if isinstance(value, dict):
        for key in ("node_ids", "nodes", "indices", "target_node_ids", "routed_node_ids", "routed_nodes", "items", "rows"):
            if key in value:
                return _coerce_node_id_sequence(value[key])
        for key in ("node_id", "index", "id"):
            if key in value:
                return [int(value[key])]
        if len(value) == 1:
            return _coerce_node_id_sequence(next(iter(value.values())))
        raise ValueError(f"Could not infer node ids from mapping keys: {sorted(value.keys())}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [int(stripped)]
    return [int(value)]


def _load_explicit_target_node_ids(path: Path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        raw = torch.load(path, map_location="cpu")
    elif suffix == ".jsonl":
        raw = _read_jsonl(path)
    elif suffix == ".json":
        raw = read_json(path, default=None)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            raw = [line.rstrip("\n") for line in handle]
    values = _coerce_node_id_sequence(raw)
    seen = set()
    ordered = []
    for item in values:
        node_id = int(item)
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    return ordered


def _load_resume_explanations(path: Path, component_name: str, quality_gate: bool = False):
    if not path or not Path(path).exists():
        return {}, {}
    by_key = {}
    rejected = defaultdict(int)
    for row in _read_jsonl(Path(path)):
        if str(row.get("component_name", component_name)) != str(component_name):
            continue
        if "node_id" not in row or "prompt_hash" not in row:
            continue
        key = (int(row["node_id"]), str(row["prompt_hash"]))
        explanation = str(row.get("explanation", "") or "").strip()
        if not explanation:
            rejected["empty"] += 1
            continue
        if quality_gate:
            issue = _explanation_quality_issue(explanation)
            if issue:
                rejected[issue] += 1
                continue
        by_key[key] = explanation
    return by_key, dict(rejected)


def _resolve_target_node_ids(args, labeled_node_count: int, full_node_count: int):
    center_node_scope_requested = str(getattr(args, "center_node_scope", "labeled")).lower()
    routed_nodes_path = getattr(args, "routed_nodes_path", None)
    target_node_source = "center_node_scope"
    raw_tweet_index_scope = center_node_scope_requested
    if routed_nodes_path:
        target_node_ids = _load_explicit_target_node_ids(Path(routed_nodes_path))
        if not target_node_ids:
            raise ValueError(f"--routed_nodes_path did not yield any node ids: {routed_nodes_path}")
        invalid = [int(item) for item in target_node_ids if int(item) < 0 or int(item) >= int(full_node_count)]
        if invalid:
            preview = invalid[:10]
            raise ValueError(
                f"--routed_nodes_path contains node ids outside [0, {int(full_node_count) - 1}]: "
                f"{preview}{' ...' if len(invalid) > len(preview) else ''}"
            )
        target_node_scope = "explicit_routed_nodes"
        target_node_source = "routed_nodes_path"
        raw_tweet_index_scope = "all_graph_nodes"
    else:
        if center_node_scope_requested == "labeled":
            target_count = int(labeled_node_count)
        elif center_node_scope_requested == "all_graph_nodes":
            target_count = int(full_node_count)
        else:
            raise ValueError(f"Unsupported center_node_scope: {center_node_scope_requested}")
        target_node_scope = center_node_scope_requested
        target_node_ids = list(range(target_count))
    if args.limit and int(args.limit) > 0:
        target_node_ids = list(target_node_ids[: int(args.limit)])
    return {
        "target_node_ids": [int(item) for item in target_node_ids],
        "target_node_scope": str(target_node_scope),
        "center_node_scope_requested": str(center_node_scope_requested),
        "target_node_source": str(target_node_source),
        "raw_tweet_index_scope": str(raw_tweet_index_scope),
        "routed_nodes_path": str(routed_nodes_path) if routed_nodes_path else "",
    }


def _scatter_selected_tensor_to_full_graph(value: torch.Tensor, target_index_tensor: torch.Tensor, full_node_count: int):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    rows = int(value.shape[0]) if value.dim() >= 1 else 0
    if rows == int(full_node_count):
        return value
    if rows != int(target_index_tensor.numel()):
        raise ValueError(
            f"Cannot scatter tensor with {rows} rows into full graph of size {full_node_count}; "
            f"expected {int(target_index_tensor.numel())} target rows."
        )
    if value.dim() == 1:
        out = torch.zeros((int(full_node_count),), dtype=value.dtype)
    elif value.dim() == 2:
        out = torch.zeros((int(full_node_count), int(value.shape[1])), dtype=value.dtype)
    else:
        raise ValueError(f"Expected 1-D or 2-D tensor for scatter, got shape {tuple(value.shape)}.")
    out[target_index_tensor] = value
    return out


def _node_text(text):
    if not isinstance(text, str):
        return ""
    return text[:2000]


def _prompt_header(classes):
    return f"Instruct: Predict the node's category from the provided context. Possible categories: {classes}."


def _format_prompt_section(name, texts):
    if not texts:
        return f"{name}: None"
    return f"{name}: " + " ".join(f"- {text}" for text in texts)


def _compact_whitespace(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\n", " ").split()).strip()


def _clean_metadata_field(value):
    return _compact_whitespace(str(value or "").replace("</s>", " "))


def _truncate_chars(text, limit):
    text = _compact_whitespace(text)
    if len(text) <= int(limit):
        return text
    return text[: max(int(limit) - 3, 0)].rstrip() + "..."


def _safe_int(value, default=0):
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return int(default)
        return int(float(text))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=0.0):
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return float(default)
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _split_norm_user_text(raw_text):
    text = str(raw_text or "")
    meta_anchor = "METADATA:"
    desc_anchor = "DESCRIPTION:"
    tweet_anchor = "TWEET:"
    meta_idx = text.find(meta_anchor)
    desc_idx = text.find(desc_anchor)
    tweet_idx = text.find(tweet_anchor)
    if meta_idx < 0:
        return "", "", ""
    meta_start = meta_idx + len(meta_anchor)
    meta_end = desc_idx if desc_idx >= 0 else (tweet_idx if tweet_idx >= 0 else len(text))
    desc_start = desc_idx + len(desc_anchor) if desc_idx >= 0 else -1
    desc_end = tweet_idx if tweet_idx >= 0 else len(text)
    tweet_start = tweet_idx + len(tweet_anchor) if tweet_idx >= 0 else -1
    metadata = text[meta_start:meta_end].strip()
    description = text[desc_start:desc_end].strip() if desc_start >= 0 else ""
    tweets = text[tweet_start:].strip() if tweet_start >= 0 else ""
    return metadata, description, tweets


def _parse_created_at(value):
    value = _compact_whitespace(value)
    if not value:
        return None
    for fmt in ("%a %b %d %H:%M:%S %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _account_age_bucket(created_at):
    created_dt = _parse_created_at(created_at)
    if created_dt is None:
        return "unknown"
    account_days = max((DEFAULT_ACCOUNT_REFERENCE_DATE - created_dt).days, 0)
    if account_days < 180:
        return "very_new"
    if account_days < 365:
        return "new"
    if account_days < 3 * 365:
        return "young"
    if account_days < 8 * 365:
        return "established"
    return "old"


def _follow_ratio_bucket(followers_count, following_count):
    ratio = float(followers_count) / max(float(following_count) + 1.0, 1.0)
    if ratio < 0.1:
        return "very_low"
    if ratio < 0.5:
        return "low"
    if ratio < 2.0:
        return "balanced"
    if ratio < 10.0:
        return "high"
    return "very_high"


def _posting_density_bucket(statuses_count, created_at):
    created_dt = _parse_created_at(created_at)
    if created_dt is None:
        return "unknown"
    account_days = max((DEFAULT_ACCOUNT_REFERENCE_DATE - created_dt).days, 1)
    density = float(statuses_count) / float(account_days)
    if density < 0.05:
        return "very_low"
    if density < 0.5:
        return "low"
    if density < 3.0:
        return "moderate"
    if density < 10.0:
        return "high"
    return "very_high"


def _tweet_length_bucket(avg_length):
    avg_length = float(avg_length)
    if avg_length <= 0.0:
        return "empty"
    if avg_length < 8.0:
        return "very_short"
    if avg_length < 20.0:
        return "short"
    if avg_length < 40.0:
        return "medium"
    return "long"


_ACCOUNT_AGE_BUCKETS = ("unknown", "very_new", "new", "established", "old")
_FOLLOW_RATIO_BUCKETS = ("very_low", "low", "balanced", "high", "very_high")
_POSTING_DENSITY_BUCKETS = ("unknown", "very_low", "low", "moderate", "high", "very_high")


def _one_hot_bucket(value, buckets):
    token = str(value or "").strip().lower()
    return [1.0 if token == item else 0.0 for item in buckets]


def _safe_log_ratio(numerator, denominator):
    return float(math.log1p(max(float(numerator), 0.0)) - math.log1p(max(float(denominator), 0.0)))


def _metadata_structured_feature_names():
    names = [
        "protected",
        "verified",
        "bio_present",
        "location_present",
        "display_name_present",
        "screen_name_present",
        "created_at_known",
        "account_days_log1p",
        "followers_log1p",
        "following_log1p",
        "listed_log1p",
        "statuses_log1p",
        "followers_following_log_ratio",
        "listed_followers_log_ratio",
        "statuses_account_days_log_ratio",
        "bio_char_len_log1p",
        "display_name_char_len_log1p",
        "screen_name_char_len_log1p",
        "screen_name_digit_ratio",
        "screen_name_underscore_ratio",
        "display_screen_name_similarity",
        "missing_profile_score",
    ]
    names.extend(f"account_age_bucket={bucket}" for bucket in _ACCOUNT_AGE_BUCKETS)
    names.extend(f"follow_ratio_bucket={bucket}" for bucket in _FOLLOW_RATIO_BUCKETS)
    names.extend(f"posting_density_bucket={bucket}" for bucket in _POSTING_DENSITY_BUCKETS)
    return names


METADATA_STRUCTURED_FEATURE_NAMES = tuple(_metadata_structured_feature_names())


def _char_ratio(text, predicate):
    text = str(text or "")
    if not text:
        return 0.0
    return float(sum(1 for ch in text if predicate(ch))) / float(len(text))


def _token_similarity(a, b):
    def _tokens(value):
        return {item for item in re.split(r"[^a-z0-9]+", str(value or "").lower()) if item}

    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return float(len(left.intersection(right))) / float(len(left.union(right)))


def _metadata_structured_features(record):
    account_days = record.get("account_days")
    account_days_f = float(account_days) if account_days is not None else 0.0
    followers_count = float(record.get("followers_count", 0) or 0)
    following_count = float(record.get("following_count", 0) or 0)
    listed_count = float(record.get("listed_count", 0) or 0)
    statuses_count = float(record.get("statuses_count", 0) or 0)
    bio = str(record.get("bio", "") or "")
    display_name = str(record.get("display_name", "") or "")
    screen_name = str(record.get("screen_name", "") or "")
    location = str(record.get("location", "") or "")
    missing_items = [
        0 if bio else 1,
        0 if display_name else 1,
        0 if screen_name else 1,
        0 if location else 1,
        0 if account_days is not None else 1,
    ]
    numeric = [
        float(bool(record.get("protected", False))),
        float(bool(record.get("verified", False))),
        float(bool(record.get("bio_present", 0))),
        float(bool(location)),
        float(bool(display_name)),
        float(bool(screen_name)),
        float(account_days is not None),
        float(math.log1p(max(account_days_f, 0.0))),
        float(math.log1p(max(followers_count, 0.0))),
        float(math.log1p(max(following_count, 0.0))),
        float(math.log1p(max(listed_count, 0.0))),
        float(math.log1p(max(statuses_count, 0.0))),
        _safe_log_ratio(followers_count, following_count),
        _safe_log_ratio(listed_count, followers_count),
        _safe_log_ratio(statuses_count, account_days_f),
        float(math.log1p(len(bio))),
        float(math.log1p(len(display_name))),
        float(math.log1p(len(screen_name))),
        _char_ratio(screen_name, str.isdigit),
        _char_ratio(screen_name, lambda ch: ch == "_"),
        _token_similarity(display_name, screen_name),
        float(sum(missing_items)) / float(len(missing_items)),
    ]
    features = (
        numeric
        + _one_hot_bucket(record.get("account_age_bucket"), _ACCOUNT_AGE_BUCKETS)
        + _one_hot_bucket(record.get("follow_ratio_bucket"), _FOLLOW_RATIO_BUCKETS)
        + _one_hot_bucket(record.get("posting_density_bucket"), _POSTING_DENSITY_BUCKETS)
    )
    if len(features) != len(METADATA_STRUCTURED_FEATURE_NAMES):
        raise RuntimeError(
            "metadata_structured feature schema mismatch: "
            f"{len(features)} values for {len(METADATA_STRUCTURED_FEATURE_NAMES)} names."
        )
    return features


def _parse_norm_user_record(raw_text, node_id):
    metadata_text, description_text, tweets_text = _split_norm_user_text(raw_text)
    metadata_fields = [_clean_metadata_field(item) for item in metadata_text.split(" </s> ")] if metadata_text else []
    while len(metadata_fields) < 10:
        metadata_fields.append("")
    created_at, location, display_name, protected, followers_count, following_count, listed_count, statuses_count, screen_name, verified = metadata_fields[:10]
    tweets = [item.strip() for item in tweets_text.split(" </s> ") if item.strip()]
    description_text = _compact_whitespace(description_text)
    created_dt = _parse_created_at(created_at)
    account_days = max((DEFAULT_ACCOUNT_REFERENCE_DATE - created_dt).days, 1) if created_dt is not None else None
    followers_count_i = _safe_int(followers_count, default=0)
    following_count_i = _safe_int(following_count, default=0)
    listed_count_i = _safe_int(listed_count, default=0)
    statuses_count_i = _safe_int(statuses_count, default=0)
    return {
        "node_id": int(node_id),
        "raw_text": str(raw_text or ""),
        "metadata_text": metadata_text,
        "description_text": description_text,
        "tweets": tweets,
        "created_at": _compact_whitespace(created_at),
        "location": _compact_whitespace(location),
        "display_name": _compact_whitespace(display_name),
        "screen_name": _compact_whitespace(screen_name),
        "protected": _safe_bool_field(protected, default=False),
        "verified": _safe_bool_field(verified, default=False),
        "followers_count": followers_count_i,
        "following_count": following_count_i,
        "listed_count": listed_count_i,
        "statuses_count": statuses_count_i,
        "bio": description_text,
        "bio_present": 1 if description_text else 0,
        "account_age_bucket": _account_age_bucket(created_at),
        "follow_ratio_bucket": _follow_ratio_bucket(followers_count_i, following_count_i),
        "posting_density_bucket": _posting_density_bucket(statuses_count_i, created_at),
        "account_days": account_days,
    }


def _format_bool(value):
    return "yes" if bool(value) else "no"


def _profile_card(record, brief=False):
    bio_limit = 120 if brief else 240
    lines = [
        f"Display name: {record['display_name'] or 'Unknown'}",
        f"Handle: @{record['screen_name']}" if record["screen_name"] else "Handle: unknown",
        f"Account created: {record['created_at'] or 'unknown'}",
        f"Location: {record['location'] or 'unknown'}",
        f"Protected account: {_format_bool(record['protected'])}",
        f"Verified: {_format_bool(record['verified'])}",
        f"Followers: {record['followers_count']}",
        f"Following: {record['following_count']}",
        f"Listed count: {record['listed_count']}",
        f"Status count: {record['statuses_count']}",
        f"Bio present: {_format_bool(record['bio_present'])}",
        f"Account age bucket: {record['account_age_bucket']}",
        f"Follow ratio bucket: {record['follow_ratio_bucket']}",
        f"Posting density bucket: {record['posting_density_bucket']}",
        f"Bio: {_truncate_chars(record['bio'], bio_limit) or 'None'}",
    ]
    return "\n".join(lines)


def _neighbor_card(record):
    return (
        f"- {record['display_name'] or 'Unknown'} (@{record['screen_name'] or 'unknown'}) | "
        f"verified={_format_bool(record['verified'])}, followers={record['followers_count']}, "
        f"following={record['following_count']}, listed={record['listed_count']}, "
        f"statuses={record['statuses_count']}, bio={_truncate_chars(record['bio'], 120) or 'None'}"
    )


def _profile_cue_summary(record, include_identity=True):
    lines = []
    if include_identity:
        lines.extend(
            [
                f"Display name: {record['display_name'] or 'Unknown'}",
                f"Handle: @{record['screen_name']}" if record["screen_name"] else "Handle: unknown",
            ]
        )
    lines.extend(
        [
            f"Verified: {_format_bool(record['verified'])}",
            f"Protected account: {_format_bool(record['protected'])}",
            f"Bio present: {_format_bool(record['bio_present'])}",
            f"Account age bucket: {record['account_age_bucket']}",
            f"Follow ratio bucket: {record['follow_ratio_bucket']}",
            f"Posting density bucket: {record['posting_density_bucket']}",
            f"Bio cue: {_truncate_chars(record['bio'], 120) or 'None'}",
        ]
    )
    return "\n".join(lines)


def _neighbor_card_v2(record):
    return (
        f"- {record['display_name'] or 'Unknown'} (@{record['screen_name'] or 'unknown'}) | "
        f"verified={_format_bool(record['verified'])}, protected={_format_bool(record['protected'])}, "
        f"bio_present={_format_bool(record['bio_present'])}, age_bucket={record['account_age_bucket']}, "
        f"follow_ratio_bucket={record['follow_ratio_bucket']}, posting_density_bucket={record['posting_density_bucket']}, "
        f"bio={_truncate_chars(record['bio'], 100) or 'None'}"
    )


def _tweet_is_retweet(text):
    text = _compact_whitespace(text)
    return text.startswith("RT @USER")


def _tweet_unique_ratio(text):
    tokens = [token for token in _compact_whitespace(text).split() if token]
    if not tokens:
        return 0.0
    return float(len(set(tokens))) / float(len(tokens))


def _tweet_sort_key(item):
    idx, tweet = item
    compact = _compact_whitespace(tweet)
    token_count = len(compact.split())
    return (
        1 if not _tweet_is_retweet(compact) else 0,
        1 if compact else 0,
        token_count,
        _tweet_unique_ratio(compact),
        -int(idx),
    )


def _sample_tweets(record, max_items=8):
    indexed = list(enumerate(record["tweets"]))
    ranked = sorted(indexed, key=_tweet_sort_key, reverse=True)
    return [tweet for _, tweet in ranked[: int(max_items)]]


def _replace_hashtag_surface(text):
    return re.sub(r"(?<!\w)#([A-Za-z0-9_]+)", r"hashtag:\1", text)


def _replace_emoji_surface(text):
    return re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
        " emoji ",
        text,
        flags=re.UNICODE,
    )


def _light_clean_tweet_text(text, keep_hashtag_surface=True, keep_emoji_surface=True):
    text = html.unescape(str(text or ""))
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", text)
    text = re.sub(r"(?<!\w)@\w+", "@USER", text)
    if keep_hashtag_surface:
        text = _replace_hashtag_surface(text)
    else:
        text = re.sub(r"(?<!\w)#\w+", "#HASHTAG", text)
    if keep_emoji_surface:
        text = _replace_emoji_surface(text)
    else:
        text = re.sub(
            r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
            " EMOJI ",
            text,
            flags=re.UNICODE,
        )
    text = text.replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).strip()


def _sample_tweets_for_explainer(record, max_items=6):
    tweets = [str(item).strip() for item in record.get("tweets", []) if str(item).strip()]
    if not tweets:
        return []
    originals = []
    retweets = []
    url_heavy = []
    hashtag_heavy = []
    repetitive = []
    for idx, tweet in enumerate(tweets):
        compact = _compact_whitespace(tweet)
        if not compact:
            continue
        token_count = len(compact.split())
        unique_ratio = _tweet_unique_ratio(compact)
        item = (idx, compact, token_count, unique_ratio)
        if _tweet_is_retweet(compact):
            retweets.append(item)
        else:
            originals.append(item)
        if "HTTPURL" in compact:
            url_heavy.append(item)
        if "hashtag:" in compact or "#HASHTAG" in compact:
            hashtag_heavy.append(item)
        if unique_ratio < 0.55:
            repetitive.append(item)

    def _rank(items):
        return sorted(items, key=lambda x: (x[2], x[3], -x[0]), reverse=True)

    picked = []
    used = set()
    buckets = [
        _rank([item for item in originals if item not in retweets]),
        _rank(retweets),
        _rank(url_heavy),
        _rank(hashtag_heavy),
        _rank(repetitive),
    ]
    for bucket in buckets:
        for item in bucket:
            if len(picked) >= int(max_items):
                break
            if item[0] in used:
                continue
            picked.append(item)
            used.add(item[0])
            break
    for item in _rank(originals + retweets):
        if len(picked) >= int(max_items):
            break
        if item[0] in used:
            continue
        picked.append(item)
        used.add(item[0])
    return [item[1] for item in sorted(picked, key=lambda x: x[0])]


def _tweet_stats_from_raw_tweets(record, max_items=6):
    tweets = [_compact_whitespace(tweet) for tweet in record["tweets"] if _compact_whitespace(tweet)]
    if not tweets:
        return {
            "sampled_tweets": [],
            "tweet_count": 0,
            "rt_ratio": 0.0,
            "url_ratio": 0.0,
            "hashtag_ratio": 0.0,
            "avg_tweet_len": 0.0,
            "avg_tweet_len_bucket": "empty",
            "duplicate_ratio": 0.0,
            "sampling_profile": {
                "original_count": 0,
                "retweet_count": 0,
                "url_heavy_count": 0,
                "hashtag_heavy_count": 0,
            },
        }
    sampled = _sample_tweets_for_explainer(record, max_items=max_items)
    tweet_count = len(tweets)
    rt_ratio = float(sum(1 for tweet in tweets if _tweet_is_retweet(tweet))) / float(tweet_count)
    url_ratio = float(sum(1 for tweet in tweets if "HTTPURL" in tweet)) / float(tweet_count)
    hashtag_ratio = float(sum(1 for tweet in tweets if "hashtag:" in tweet or "#HASHTAG" in tweet)) / float(tweet_count)
    lengths = [len(tweet.split()) for tweet in tweets]
    avg_tweet_len = float(np.mean(lengths)) if lengths else 0.0
    duplicate_ratio = 1.0 - float(len(set(tweets))) / float(tweet_count)
    return {
        "sampled_tweets": sampled,
        "tweet_count": tweet_count,
        "rt_ratio": rt_ratio,
        "url_ratio": url_ratio,
        "hashtag_ratio": hashtag_ratio,
        "avg_tweet_len": avg_tweet_len,
        "avg_tweet_len_bucket": _tweet_length_bucket(avg_tweet_len),
        "duplicate_ratio": duplicate_ratio,
        "sampling_profile": {
            "original_count": int(sum(1 for tweet in sampled if not _tweet_is_retweet(tweet))),
            "retweet_count": int(sum(1 for tweet in sampled if _tweet_is_retweet(tweet))),
            "url_heavy_count": int(sum(1 for tweet in sampled if "HTTPURL" in tweet)),
            "hashtag_heavy_count": int(sum(1 for tweet in sampled if "hashtag:" in tweet or "#HASHTAG" in tweet)),
        },
    }


def _build_raw_tweet_record_for_user(user_id, user_profile, tweet_ids, tweet_text_by_id, clean_level, sample_size, keep_hashtag_surface=True, keep_emoji_surface=True):
    profile = dict(user_profile or {})
    public_metrics = dict(profile.get("public_metrics") or {})
    created_at = _compact_whitespace(profile.get("created_at"))
    location = _compact_whitespace(profile.get("location"))
    display_name = _compact_whitespace(profile.get("name"))
    screen_name = _compact_whitespace(profile.get("username"))
    bio = _compact_whitespace(profile.get("description"))
    if clean_level == "norm_compatible":
        clean_tweets = [_compact_text(tweet_text_by_id[tweet_id]) for tweet_id in tweet_ids if tweet_id in tweet_text_by_id]
    else:
        clean_tweets = [
            _light_clean_tweet_text(
                tweet_text_by_id[tweet_id],
                keep_hashtag_surface=keep_hashtag_surface,
                keep_emoji_surface=keep_emoji_surface,
            )
            for tweet_id in tweet_ids
            if tweet_id in tweet_text_by_id
        ]
    clean_tweets = [tweet for tweet in clean_tweets if tweet]
    followers_count = _safe_int(public_metrics.get("followers_count"), default=0)
    following_count = _safe_int(public_metrics.get("following_count"), default=0)
    listed_count = _safe_int(public_metrics.get("listed_count"), default=0)
    statuses_count = _safe_int(public_metrics.get("tweet_count"), default=len(clean_tweets))
    record = {
        "node_id": int(_safe_int(str(user_id).lstrip("u"), default=0)),
        "source_user_id": str(user_id),
        "display_name": display_name,
        "screen_name": screen_name,
        "created_at": created_at,
        "location": location,
        "protected": _safe_bool_field(profile.get("protected"), default=False),
        "verified": _safe_bool_field(profile.get("verified"), default=False),
        "followers_count": followers_count,
        "following_count": following_count,
        "listed_count": listed_count,
        "statuses_count": statuses_count,
        "bio": bio,
        "bio_present": 1 if bio else 0,
        "account_age_bucket": _account_age_bucket(created_at),
        "follow_ratio_bucket": _follow_ratio_bucket(followers_count, following_count),
        "posting_density_bucket": _posting_density_bucket(statuses_count, created_at),
        "tweets": clean_tweets,
        "raw_tweet_ids": list(tweet_ids),
        "raw_tweet_count": int(len(tweet_ids)),
    }
    record["tweet_stats"] = _tweet_stats_from_raw_tweets(record, max_items=sample_size)
    return record


def _tweet_stats(record):
    tweets = [_compact_whitespace(tweet) for tweet in record["tweets"] if _compact_whitespace(tweet)]
    if not tweets:
        return {
            "sampled_tweets": [],
            "tweet_count": 0,
            "rt_ratio": 0.0,
            "url_ratio": 0.0,
            "hashtag_ratio": 0.0,
            "avg_tweet_len": 0.0,
            "avg_tweet_len_bucket": "empty",
            "duplicate_ratio": 0.0,
        }
    sampled = _sample_tweets(record, max_items=8)
    tweet_count = len(tweets)
    rt_ratio = float(sum(1 for tweet in tweets if _tweet_is_retweet(tweet))) / float(tweet_count)
    url_ratio = float(sum(1 for tweet in tweets if "HTTPURL" in tweet)) / float(tweet_count)
    hashtag_ratio = float(sum(1 for tweet in tweets if "#HASHTAG" in tweet)) / float(tweet_count)
    lengths = [len(tweet.split()) for tweet in tweets]
    avg_tweet_len = float(np.mean(lengths)) if lengths else 0.0
    duplicate_ratio = 1.0 - float(len(set(tweets))) / float(tweet_count)
    return {
        "sampled_tweets": sampled,
        "tweet_count": tweet_count,
        "rt_ratio": rt_ratio,
        "url_ratio": url_ratio,
        "hashtag_ratio": hashtag_ratio,
        "avg_tweet_len": avg_tweet_len,
        "avg_tweet_len_bucket": _tweet_length_bucket(avg_tweet_len),
        "duplicate_ratio": duplicate_ratio,
    }


def _get_tweet_stats(record, sample_size=8):
    cached = record.get("tweet_stats")
    if isinstance(cached, dict):
        return cached
    if record.get("tweet_source") == "raw_post_edges":
        stats = _tweet_stats_from_raw_tweets(record, max_items=sample_size)
    else:
        stats = _tweet_stats(record)
    record["tweet_stats"] = stats
    return stats


def _tweet_behavior_summary(record, tweet_stats):
    return "\n".join(
        [
            f"Tweet sample count: {len(tweet_stats['sampled_tweets'])}",
            f"Total non-empty tweets: {tweet_stats['tweet_count']}",
            f"Retweet ratio: {tweet_stats['rt_ratio']:.2f}",
            f"URL ratio: {tweet_stats['url_ratio']:.2f}",
            f"Hashtag ratio: {tweet_stats['hashtag_ratio']:.2f}",
            f"Average tweet length bucket: {tweet_stats['avg_tweet_len_bucket']}",
            f"Duplicate ratio: {tweet_stats['duplicate_ratio']:.2f}",
            f"Posting density bucket: {record['posting_density_bucket']}",
        ]
    )


def _tweet_samples_block(tweet_stats):
    if not tweet_stats["sampled_tweets"]:
        return "TWEET_SAMPLES:\n- None"
    lines = ["TWEET_SAMPLES:"]
    for tweet in tweet_stats["sampled_tweets"]:
        lines.append(f"- {_truncate_chars(tweet, 260)}")
    return "\n".join(lines)


def _deterministic_ego_explain_fallback(record):
    return " ".join(
        [
            f"The account uses the display name {record['display_name'] or 'unknown'} and the handle @{record['screen_name'] or 'unknown'}.",
            f"It is verified={_format_bool(record['verified'])}, protected={_format_bool(record['protected'])}, and has followers={record['followers_count']} and following={record['following_count']}.",
            f"The account age bucket is {record['account_age_bucket']} and the posting density bucket is {record['posting_density_bucket']}.",
            f"The bio is {_truncate_chars(record['bio'], 160) or 'missing'}, which provides additional profile context.",
        ]
    )


def _deterministic_tweet_explain_fallback(record, tweet_stats):
    return " ".join(
        [
            f"The account shows a posting density bucket of {record['posting_density_bucket']} and a tweet count of {tweet_stats['tweet_count']}.",
            f"Its retweet ratio is {tweet_stats['rt_ratio']:.2f}, url ratio is {tweet_stats['url_ratio']:.2f}, and hashtag ratio is {tweet_stats['hashtag_ratio']:.2f}.",
            f"The average tweet length bucket is {tweet_stats['avg_tweet_len_bucket']} with duplicate ratio {tweet_stats['duplicate_ratio']:.2f}.",
            "These cues summarize the account's observable posting style and content behavior.",
        ]
    )


def _deterministic_graph_explain_fallback(direction_name, total_count, reciprocal_ratio_selected, neighbor_records):
    label = "follows" if direction_name == "following" else "is followed by"
    top_neighbor = neighbor_records[0] if neighbor_records else None
    neighbor_text = (
        f"The strongest visible neighbor is {top_neighbor['display_name'] or 'unknown'} (@{top_neighbor['screen_name'] or 'unknown'})."
        if top_neighbor is not None
        else "No informative directed neighbors are available."
    )
    return " ".join(
        [
            f"The account {label} {int(total_count)} directed neighbors in this view.",
            f"The selected reciprocal ratio is {float(reciprocal_ratio_selected):.2f}.",
            neighbor_text,
            "This summary captures the directed neighborhood evidence in this view.",
        ]
    )


def _deterministic_conflict_explain_fallback(row):
    hints = row.get("conflict_mismatch_hints", []) or []
    if hints:
        hint_text = " ".join(item.lstrip("- ").strip() for item in hints[:3])
    else:
        hint_text = "The profile, posting behavior, and social neighborhood do not expose a strong explicit mismatch cue."
    return " ".join(
        [
            "Cross-view evidence can be summarized by comparing profile cues, posting behavior, and directed neighborhood role.",
            hint_text,
            "This summary preserves the main agreements, tensions, and missing evidence across views.",
        ]
    )


def _summary_generation_system():
    return (
        "You summarize social-media account evidence. "
        "Be concise, balanced, and faithful to the provided evidence."
    )


def _summary_prompt_rules():
    return [
        "Write 3-4 short evidence-grounded sentences.",
        "Summarize both organic-looking and automation-looking cues when they are visible.",
        "If the evidence is weak, mixed, or missing, say so explicitly.",
        "Do not output a final human/bot label, confidence score, or recommendation.",
    ]


def _expert_embedding_prompt(component_name, explanation_text):
    label_map = {
        "ego": "profile-based",
        "graph_following": "following-side social-role",
        "graph_follower": "follower-side audience",
        "tweet": "tweet-behavior",
        "conflict": "cross-view conflict",
    }
    explanation_text = _compact_whitespace(explanation_text)
    component_label = label_map.get(component_name, component_name.replace("_", " "))
    return (
        f"Instruct: Encode the {component_label} evidence summary for downstream reasoning.\n"
        f"Query: [SUMMARY] {explanation_text} </END>"
    )


def _tweet_explain_generation_prompt(record, tweet_stats):
    system = _summary_generation_system()
    user = "\n".join(
        [
            "Summarize this account's posting behavior from the provided tweet evidence.",
            "Focus on recurring content patterns, interaction style, topical consistency, and any clear anomalies in the sampled tweets.",
            *_summary_prompt_rules(),
            "PROFILE_CUE:",
            _profile_cue_summary(record, include_identity=True),
            "TWEET_BEHAVIOR_SUMMARY:",
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
        ]
    )
    return {
        "system": system,
        "user": user,
        "fallback_explanation": _deterministic_tweet_explain_fallback(record, tweet_stats),
        "prompt_role": "tweet_explainer",
    }


def _build_raw_tweet_source_bundle(args, dataset_path: Path, graph_data_variant: str, center_node_scope: str, target_node_ids):
    node_source_path = Path(getattr(args, "node_source_path", None) or _resolve_default_node_source_path(dataset_path))
    edge_source_path = Path(getattr(args, "edge_source_path", None) or _resolve_default_edge_source_path(dataset_path))
    if not node_source_path.exists():
        raise FileNotFoundError(f"raw_post_edges requires node source at {node_source_path}")
    if not edge_source_path.exists():
        raise FileNotFoundError(f"raw_post_edges requires edge source at {edge_source_path}")
    index_to_user_id = _build_index_to_user_id(
        dataset_path,
        graph_data_variant=graph_data_variant,
        center_node_scope=center_node_scope,
        target_node_ids=target_node_ids,
    )
    target_user_ids = [index_to_user_id[int(idx)] for idx in target_node_ids]
    node_indices = _build_raw_node_indices(node_source_path, target_user_ids=target_user_ids, target_tweet_ids=None, need_tweets=False)
    user_profile_by_id = node_indices["user_profile_by_id"]
    direction = _resolve_post_edge_direction(
        edge_source_path,
        known_user_ids=user_profile_by_id.keys(),
        known_tweet_ids=None,
    )
    user_tweet_map = _build_user_tweet_map(
        edge_source_path,
        target_user_ids=user_profile_by_id.keys(),
        target_tweet_ids=None,
        direction=direction,
    )
    target_tweet_ids = set()
    for tweet_ids in user_tweet_map.values():
        target_tweet_ids.update(str(item) for item in tweet_ids)
    tweet_only_indices = _build_raw_node_indices(
        node_source_path,
        target_user_ids=None,
        target_tweet_ids=target_tweet_ids,
        need_tweets=True,
    )
    tweet_text_by_id = tweet_only_indices["tweet_text_by_id"]
    user_tweet_map = {
        user_id: [tweet_id for tweet_id in tweet_ids if tweet_id in tweet_text_by_id]
        for user_id, tweet_ids in user_tweet_map.items()
    }
    return {
        "node_source_path": str(node_source_path),
        "edge_source_path": str(edge_source_path),
        "post_edge_direction": direction,
        "index_to_user_id": index_to_user_id,
        "user_profile_by_id": user_profile_by_id,
        "tweet_text_by_id": tweet_text_by_id,
        "user_tweet_map": user_tweet_map,
    }


def _attach_raw_tweet_records(records, raw_tweet_bundle, args):
    enriched = []
    fallback_count = 0
    sample_size = int(getattr(args, "tweet_sample_size", 6))
    clean_level = str(getattr(args, "tweet_clean_level", "light")).lower()
    keep_hashtag_surface = bool(getattr(args, "tweet_keep_hashtag_surface", True))
    keep_emoji_surface = bool(getattr(args, "tweet_keep_emoji_surface", True))
    user_profile_by_id = raw_tweet_bundle["user_profile_by_id"]
    tweet_text_by_id = raw_tweet_bundle["tweet_text_by_id"]
    user_tweet_map = raw_tweet_bundle["user_tweet_map"]
    index_to_user_id = raw_tweet_bundle["index_to_user_id"]
    covered = 0
    for record in records:
        record = dict(record)
        node_idx = int(record["node_id"])
        user_id = index_to_user_id.get(node_idx)
        raw_profile = user_profile_by_id.get(user_id) if user_id else None
        raw_tweet_ids = user_tweet_map.get(user_id, []) if user_id else []
        usable_tweet_ids = [tweet_id for tweet_id in raw_tweet_ids if tweet_id in tweet_text_by_id]
        if raw_profile and usable_tweet_ids:
            raw_record = _build_raw_tweet_record_for_user(
                user_id,
                raw_profile,
                usable_tweet_ids,
                tweet_text_by_id,
                clean_level=clean_level,
                sample_size=sample_size,
                keep_hashtag_surface=keep_hashtag_surface,
                keep_emoji_surface=keep_emoji_surface,
            )
            for key in (
                "tweets",
                "display_name",
                "screen_name",
                "created_at",
                "location",
                "protected",
                "verified",
                "followers_count",
                "following_count",
                "listed_count",
                "statuses_count",
                "bio",
                "bio_present",
                "account_age_bucket",
                "follow_ratio_bucket",
                "posting_density_bucket",
                "tweet_stats",
            ):
                record[key] = raw_record[key]
            record["tweet_source"] = "raw_post_edges"
            record["raw_tweet_ids"] = raw_record["raw_tweet_ids"]
            record["raw_tweet_count"] = raw_record["raw_tweet_count"]
            covered += 1
        else:
            record["tweet_source"] = "norm_user_text_fallback"
            record["raw_tweet_ids"] = []
            record["raw_tweet_count"] = 0
            record["tweet_stats"] = _get_tweet_stats(record, sample_size=sample_size)
            fallback_count += 1
        enriched.append(record)
    return enriched, {
        "raw_tweet_user_coverage": float(covered / max(len(records), 1)),
        "raw_tweet_fallback_count": int(fallback_count),
        "raw_tweet_covered_count": int(covered),
    }


def _graph_explain_generation_prompt(direction_name, ego_record, neighbor_records, summary):
    if direction_name == "following":
        task_line = (
            "Summarize who this account chooses to follow in this directed view. "
            "Describe the dominant neighbor themes, notable support/contrast examples, and whether the view looks coherent, mixed, or sparse."
        )
        heading = "FOLLOWING_NEIGHBORS:"
    else:
        task_line = (
            "Summarize who follows this account in this directed view. "
            "Describe the dominant audience themes, notable support/contrast examples, and whether the view looks coherent, mixed, or sparse."
        )
        heading = "FOLLOWER_NEIGHBORS:"
    user = "\n".join(
        [
            task_line,
            *_summary_prompt_rules(),
            "EGO_PROFILE_CUE:",
            _profile_cue_summary(ego_record, include_identity=True),
            "DIRECTIONAL_SUMMARY:",
            f"count_{direction_name}: {int(summary['count'])}",
            f"has_{direction_name}: {_format_bool(summary['count'] > 0)}",
            f"selected_{direction_name}: {int(summary['selected_count'])}",
            f"reciprocal_ratio_{direction_name}_selected: {float(summary['reciprocal_ratio_selected']):.2f}",
            heading,
            *([_neighbor_card_v2(item) for item in neighbor_records] or ["- None"]),
        ]
    )
    return {
        "system": _summary_generation_system(),
        "user": user,
        "fallback_explanation": _deterministic_graph_explain_fallback(
            direction_name,
            summary["count"],
            summary["reciprocal_ratio_selected"],
            neighbor_records,
        ),
        "prompt_role": f"{direction_name}_graph_explainer",
    }


def _conflict_explain_generation_prompt(row):
    mismatch_hints = list(row.get("conflict_mismatch_hints", []) or [])
    user = "\n".join(
        [
            "Summarize the main consistencies, tensions, and unresolved gaps across the profile, tweet, following, and follower views.",
            "When one view is weak because evidence is sparse or noisy, state that limitation instead of resolving the label.",
            *_summary_prompt_rules(),
            "PROFILE_CUE_SUMMARY:",
            row.get("profile_cue_summary", ""),
            "TWEET_EXPERT_SUMMARY:",
            row.get("tweet_explanation", "None"),
            "FOLLOWING_EXPERT_SUMMARY:",
            row.get("graph_following_explanation", "None"),
            "FOLLOWER_EXPERT_SUMMARY:",
            row.get("graph_follower_explanation", "None"),
            "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
        ]
    )
    return {
        "system": _summary_generation_system(),
        "user": user,
        "fallback_explanation": _deterministic_conflict_explain_fallback(row),
        "prompt_role": "conflict_explainer",
    }


def _hf_cache_root():
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _looks_like_pretrained_dir(path: Path):
    path = Path(path)
    if not path.is_dir():
        return False
    marker_files = (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / marker).exists() for marker in marker_files)


def _ordered_snapshot_dirs(cache_dir: Path):
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
        snapshot_text = str(snapshot_dir)
        if snapshot_text in seen:
            continue
        seen.add(snapshot_text)
        result.append(snapshot_dir)
    return result


def _local_pretrained_dir_candidates(path: Path):
    root = Path(path).expanduser()
    candidates = []
    seen = set()

    def add(candidate):
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            return
        seen.add(key)
        if _looks_like_pretrained_dir(candidate):
            candidates.append(candidate)

    add(root)
    for snapshot_dir in _ordered_snapshot_dirs(root):
        add(snapshot_dir)
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            add(child)
            for snapshot_dir in _ordered_snapshot_dirs(child):
                add(snapshot_dir)
    return candidates


def _resolve_local_pretrained_source(model_source):
    text = str(model_source or "").strip()
    text = _MODEL_SOURCE_ALIASES.get(text.lower(), text)
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        candidates = _local_pretrained_dir_candidates(candidate)
        if len(candidates) == 1:
            return str(candidates[0])
        if len(candidates) > 1:
            raise ValueError(
                f"Model source {model_source!r} contains multiple local pretrained candidates. "
                "Pass the exact snapshot/model directory instead: "
                + ", ".join(str(item) for item in candidates[:5])
                + (" ..." if len(candidates) > 5 else "")
            )
        return None
    normalized = text.replace("\\", "/").strip("/")
    if not normalized or "/" not in normalized:
        return None
    cache_dir = _hf_cache_root() / f"models--{normalized.replace('/', '--')}"
    for snapshot_dir in _ordered_snapshot_dirs(cache_dir):
        if _looks_like_pretrained_dir(snapshot_dir):
            return str(snapshot_dir)
    return None


def _require_local_pretrained_source(model_source, *, model_role):
    resolved = _resolve_local_pretrained_source(model_source)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"{model_role} source {model_source!r} is not available as a local path or local HuggingFace cache snapshot. "
        "Precompute runs in offline-first mode; provide a local snapshot path or pre-cache the model before running."
    )


def _infer_embedding_encoder_tag(model_source):
    token = str(model_source or "").replace("\\", "/").lower()
    if "roberta-finetuned" in token or "roberta_finetuned" in token or "roberta-f" in token:
        return "roberta_finetuned"
    if "xlm-roberta" in token:
        return "xlm_roberta"
    if "roberta" in token:
        return "roberta"
    if "qwen" in token:
        return "qwen3"
    if "bert" in token:
        return "bert"
    return "encoder"


def _resolve_finetuned_roberta_checkpoint(args):
    explicit = getattr(args, "finetuned_roberta_checkpoint_path", None)
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"--finetuned_roberta_checkpoint_path does not exist: {path}")
    seed = int(getattr(args, "seed", 1))
    dataset = str(getattr(args, "dataset", "TwiBot-20"))
    candidates = []
    for root in (Path.cwd(), Path.cwd().parent, Path("/root/workspace/LMbot")):
        candidates.append(root / f"{dataset}_seed_{seed}" / "checkpoints" / "LM_pretrain" / "best.pkl")
        candidates.append(root / f"{dataset}_seed_{seed}" / "checkpoints" / "LM" / "best.pkl")
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


def _resolve_pooling_mode(model_source, model):
    token = str(model_source or "").replace("\\", "/").lower()
    config = getattr(model, "config", None)
    if "qwen" in token or bool(getattr(config, "is_decoder", False)):
        return "last_token"
    if _infer_embedding_encoder_tag(model_source) == "roberta_finetuned":
        return "simteg_mean"
    return "masked_mean"


def _is_prompt_expert_v2(args):
    return str(getattr(args, "prompt_family_version", "v1")).lower() == "v2" and str(
        getattr(args, "prompt_mode", "")
    ).startswith("expert_")


def _has_cli_option(raw_argv, option_name):
    prefix = f"{option_name}="
    return any(item == option_name or str(item).startswith(prefix) for item in raw_argv)


def _apply_prompt_expert_v2_encoder_defaults(args, raw_argv=None):
    raw_argv = list(sys.argv[1:] if raw_argv is None else raw_argv)
    if not _is_prompt_expert_v2(args):
        return args
    if not _has_cli_option(raw_argv, "--model_path") and str(getattr(args, "model_path", "")) == DEFAULT_QWEN_MODEL_PATH:
        args.model_path = DEFAULT_FINETUNED_ROBERTA_MODEL_ALIAS
    if not _has_cli_option(raw_argv, "--max_length_hop"):
        args.max_length_hop = 512
    if not _has_cli_option(raw_argv, "--max_length_ego"):
        args.max_length_ego = 512
    if not _has_cli_option(raw_argv, "--normalize"):
        args.normalize = False
    return args


def _model_context_limit(tokenizer, model):
    candidates = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        candidates.append(int(tokenizer_limit))
    config = getattr(model, "config", None)
    max_position_embeddings = getattr(config, "max_position_embeddings", None)
    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
        candidates.append(int(max_position_embeddings))
    return min(candidates) if candidates else None


def _effective_embedding_max_length(tokenizer, model, requested_max_length):
    requested = max(int(requested_max_length), 1)
    context_limit = _model_context_limit(tokenizer, model)
    if context_limit is None:
        return requested
    return min(requested, int(context_limit))


def _fallback_explanation_text(row):
    text = str(row.get("fallback_explanation", "") or "").strip()
    if text:
        return text
    profile_card = row.get("profile_card", "")
    if profile_card:
        return " ".join(
            [
                "The profile provides limited generated summary output.",
                _compact_whitespace(profile_card)[:320],
            ]
        ).strip()
    return "The profile provides limited generated summary output."


def _deterministic_explanations(generation_rows):
    return [_fallback_explanation_text(row) for row in generation_rows]


def _make_glance_prompts(ego_text, hop1_texts, hop2_texts, classes):
    header = _prompt_header(classes)
    ego_only = f"{header} Query: EGO: {ego_text} Category? </END>"
    hop1 = f"{header} Query: EGO: {ego_text} {_format_prompt_section('HOP1', hop1_texts)} Category? </END>"
    hop2 = f"{header} Query: EGO: {ego_text} {_format_prompt_section('HOP2', hop2_texts)} Category? </END>"
    return ego_only, hop1, hop2


def _make_relation_aware_prompts(ego_text, following_texts, follower_texts, classes):
    header = _prompt_header(classes)
    ego_only = f"{header} Query: EGO: {ego_text} Category? </END>"
    one_hop = (
        f"{header} Query: EGO: {ego_text} "
        f"{_format_prompt_section('FOLLOWING', following_texts)} "
        f"{_format_prompt_section('FOLLOWER', follower_texts)} Category? </END>"
    )
    return ego_only, one_hop


def _build_graph_context(edge_index, edge_type, num_nodes):
    edge_index = edge_index.detach().cpu().long()
    edge_type = edge_type.detach().cpu().long().view(-1)
    src = edge_index[0].numpy().astype(np.int64)
    dst = edge_index[1].numpy().astype(np.int64)
    rel = edge_type.numpy().astype(np.int64)
    incoming = [[] for _ in range(num_nodes)]
    following = [[] for _ in range(num_nodes)]
    follower = [[] for _ in range(num_nodes)]
    undirected = [set() for _ in range(num_nodes)]
    for s, d, r in zip(src.tolist(), dst.tolist(), rel.tolist()):
        if not (0 <= s < num_nodes and 0 <= d < num_nodes):
            continue
        incoming[d].append(s)
        undirected[s].add(d)
        undirected[d].add(s)
        if int(r) == 1:
            following[s].append(d)
        elif int(r) == 0:
            follower[d].append(s)
    return {
        "incoming": incoming,
        "following": following,
        "follower": follower,
        "undirected": undirected,
    }


def _sample_neighbors_uniform(nodes, cap, seed, node_id):
    unique = sorted(set(int(n) for n in nodes if int(n) >= 0))
    if len(unique) <= cap:
        return unique
    rng = np.random.default_rng(int(seed) + int(node_id))
    sampled = rng.choice(unique, size=int(cap), replace=False)
    return sorted(int(x) for x in sampled.tolist())


def _relation_heuristic_sort_key(node_id, candidate, texts, context):
    following = context["following"]
    follower = context["follower"]
    undirected = context["undirected"]
    text_len = len(texts[candidate]) if isinstance(texts[candidate], str) else 0
    has_text = 1 if text_len > 0 else 0
    ego_follows_candidate = (candidate in following[node_id]) or (node_id in follower[candidate])
    candidate_follows_ego = (candidate in follower[node_id]) or (node_id in following[candidate])
    mutual = 1 if (ego_follows_candidate and candidate_follows_ego) else 0
    common_neighbors = len(undirected[node_id].intersection(undirected[candidate]))
    total_degree = len(undirected[candidate])
    score = (
        2.0 * has_text
        + 1.5 * mutual
        + 1.0 * min(common_neighbors, 3)
        + 0.5 * min(math.log1p(total_degree), 4.0)
        + 0.5 * min(text_len / 400.0, 5.0)
    )
    return (-score, -has_text, -mutual, -common_neighbors, -total_degree, -text_len, int(candidate))


def _rank_directional_candidates(node_id, nodes, texts, context):
    unique = sorted(set(int(n) for n in nodes if int(n) >= 0 and int(n) != int(node_id)))
    return sorted(unique, key=lambda cand: _relation_heuristic_sort_key(node_id, cand, texts, context))


def _center_induced_relation_stats(node_id, candidate, texts, context, selection_features):
    following = context["following"]
    follower = context["follower"]
    undirected = context["undirected"]
    text_len = len(texts[candidate]) if isinstance(texts[candidate], str) else 0
    has_text = 1 if text_len > 0 else 0
    ego_follows_candidate = (candidate in following[node_id]) or (node_id in follower[candidate])
    candidate_follows_ego = (candidate in follower[node_id]) or (node_id in following[candidate])
    mutual = 1 if (ego_follows_candidate and candidate_follows_ego) else 0
    common_neighbors = len(undirected[node_id].intersection(undirected[candidate]))
    candidate_degree = len(undirected[candidate])
    center_vec = selection_features[int(node_id)]
    candidate_vec = selection_features[int(candidate)]
    similarity = float(torch.sum(center_vec * candidate_vec).item())
    nontrivial_structure = 1 if (candidate_degree > 1 or common_neighbors > 0 or mutual > 0) else 0
    return {
        "candidate": int(candidate),
        "similarity": similarity,
        "mutual": int(mutual),
        "common_neighbors": int(common_neighbors),
        "candidate_degree": int(candidate_degree),
        "text_length": int(text_len),
        "has_text": int(has_text),
        "nontrivial_structure": int(nontrivial_structure),
        "reciprocal": int(mutual),
    }


def _center_support_sort_key(stats):
    return (
        -float(stats["similarity"]),
        -int(stats["mutual"]),
        -int(stats["common_neighbors"]),
        -int(stats["candidate_degree"]),
        -int(stats["text_length"]),
        int(stats["candidate"]),
    )


def _center_contrast_sort_key(stats):
    return (
        float(stats["similarity"]),
        -int(stats["has_text"]),
        -int(stats["nontrivial_structure"]),
        -int(stats["candidate_degree"]),
        -int(stats["common_neighbors"]),
        -int(stats["text_length"]),
        int(stats["candidate"]),
    )


def _partition_center_induced_candidates(candidate_stats, quota):
    quota = max(int(quota), 0)
    support_quota = int(math.ceil(float(quota) / 2.0))
    contrast_quota = int(math.floor(float(quota) / 2.0))
    support_ranked = sorted(candidate_stats, key=_center_support_sort_key)
    selected_support = support_ranked[:support_quota]
    used = {int(item["candidate"]) for item in selected_support}
    contrast_ranked = sorted(
        [item for item in candidate_stats if int(item["candidate"]) not in used],
        key=_center_contrast_sort_key,
    )
    selected_contrast = contrast_ranked[:contrast_quota]
    used.update(int(item["candidate"]) for item in selected_contrast)
    fallback_used = False
    if len(selected_support) + len(selected_contrast) < quota:
        fallback_used = True
        support_overflow = [item for item in support_ranked if int(item["candidate"]) not in used]
        contrast_overflow = [item for item in contrast_ranked if int(item["candidate"]) not in used]
        overflow = support_overflow + contrast_overflow
        for item in overflow:
            if len(selected_support) + len(selected_contrast) >= quota:
                break
            if int(item["candidate"]) in used:
                continue
            if len(selected_support) < support_quota:
                selected_support.append(item)
            else:
                selected_contrast.append(item)
            used.add(int(item["candidate"]))
    return selected_support, selected_contrast, fallback_used


def _selected_similarity_mean(items):
    if not items:
        return 0.0
    return float(np.mean([float(item["similarity"]) for item in items]))


def _selected_reciprocal_ratio(items):
    if not items:
        return 0.0
    return float(np.mean([float(item["reciprocal"]) for item in items]))


def _select_center_induced_directional_neighbors(
    node_id,
    texts,
    context,
    selection_features,
    following_quota,
    follower_quota,
):
    following_candidates = [
        _center_induced_relation_stats(node_id, cand, texts, context, selection_features)
        for cand in sorted(set(int(n) for n in context["following"][node_id] if int(n) >= 0 and int(n) != int(node_id)))
    ]
    follower_candidates = [
        _center_induced_relation_stats(node_id, cand, texts, context, selection_features)
        for cand in sorted(set(int(n) for n in context["follower"][node_id] if int(n) >= 0 and int(n) != int(node_id)))
    ]

    following_support, following_contrast, following_fallback = _partition_center_induced_candidates(
        following_candidates,
        quota=int(following_quota),
    )
    follower_support, follower_contrast, follower_fallback = _partition_center_induced_candidates(
        follower_candidates,
        quota=int(follower_quota),
    )

    summary = {
        "candidate_count_following": int(len(following_candidates)),
        "candidate_count_follower": int(len(follower_candidates)),
        "selected_count_following": int(len(following_support) + len(following_contrast)),
        "selected_count_follower": int(len(follower_support) + len(follower_contrast)),
        "mean_sim_following_support": _selected_similarity_mean(following_support),
        "mean_sim_following_contrast": _selected_similarity_mean(following_contrast),
        "mean_sim_follower_support": _selected_similarity_mean(follower_support),
        "mean_sim_follower_contrast": _selected_similarity_mean(follower_contrast),
        "reciprocal_ratio_following_selected": _selected_reciprocal_ratio(following_support + following_contrast),
        "reciprocal_ratio_follower_selected": _selected_reciprocal_ratio(follower_support + follower_contrast),
        "fallback_used": bool(following_fallback or follower_fallback),
    }
    return {
        "following_support": following_support,
        "following_contrast": following_contrast,
        "follower_support": follower_support,
        "follower_contrast": follower_contrast,
        "summary": summary,
    }


def _sample_directional_neighbors(node_id, texts, context, following_quota, follower_quota):
    following_ranked = _rank_directional_candidates(node_id, context["following"][node_id], texts, context)
    follower_ranked = _rank_directional_candidates(node_id, context["follower"][node_id], texts, context)
    selected_following = following_ranked[: int(following_quota)]
    selected_follower = follower_ranked[: int(follower_quota)]
    total_budget = int(following_quota) + int(follower_quota)
    if len(selected_following) + len(selected_follower) >= total_budget:
        return selected_following, selected_follower
    overflow = []
    overflow.extend(following_ranked[len(selected_following) :])
    overflow.extend(follower_ranked[len(selected_follower) :])
    overflow = sorted(set(overflow), key=lambda cand: _relation_heuristic_sort_key(node_id, cand, texts, context))
    used = set(selected_following) | set(selected_follower)
    while len(selected_following) + len(selected_follower) < total_budget and overflow:
        candidate = overflow.pop(0)
        if candidate in used:
            continue
        if candidate in context["following"][node_id]:
            selected_following.append(candidate)
        else:
            selected_follower.append(candidate)
        used.add(candidate)
    return selected_following, selected_follower


def _build_glance_prompt_bundle(texts, edge_index, edge_type, cap, seed, classes, target_node_ids=None):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    prompts = {"ego": [], "hop1": [], "hop2": []}
    counts = {"ego_nodes": [], "hop1_nodes": [], "hop2_nodes": []}
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    for node_id in target_ids:
        raw_text = texts[node_id]
        ego_text = _node_text(raw_text)
        hop1_nodes = _sample_neighbors_uniform(context["incoming"][node_id], cap, seed, node_id)
        hop2_nodes = []
        for neigh in hop1_nodes:
            hop2_nodes.extend(context["incoming"][neigh])
        hop2_nodes = _sample_neighbors_uniform(
            [n for n in hop2_nodes if n != node_id and n not in hop1_nodes],
            cap * cap,
            seed,
            node_id,
        )
        hop1_texts = [_node_text(texts[n]) for n in hop1_nodes[:cap]]
        hop2_texts = [_node_text(texts[n]) for n in hop2_nodes[: cap * cap]]
        ego_prompt, hop1_prompt, hop2_prompt = _make_glance_prompts(ego_text, hop1_texts, hop2_texts, classes)
        prompts["ego"].append(ego_prompt)
        prompts["hop1"].append(hop1_prompt)
        prompts["hop2"].append(hop2_prompt)
        counts["ego_nodes"].append(1)
        counts["hop1_nodes"].append(1 + len(hop1_texts))
        counts["hop2_nodes"].append(1 + len(hop2_texts))
    return {
        "prompt_family": "glance_legacy",
        "prompt_style": "instruction_query",
        "prompt_components": prompts,
        "component_max_length_group": {"ego": "ego", "hop1": "hop", "hop2": "hop"},
        "counts": counts,
        "neighbor_sample_policy": "uniform_without_replacement_seeded_by_node",
        "component_order": {
            "glance_ego": ["ego"],
            "glance_hop1": ["hop1"],
            "glance_hop2": ["hop2"],
            "glance_concat_ego_hop1_hop2": ["ego", "hop1", "hop2"],
        },
    }


def _build_relation_aware_prompt_bundle(texts, edge_index, edge_type, following_quota, follower_quota, classes, target_node_ids=None):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    prompts = {"ego": [], "directional_1hop": []}
    counts = {
        "ego_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "directional_1hop_nodes": [],
    }
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    for node_id in target_ids:
        raw_text = texts[node_id]
        ego_text = _node_text(raw_text)
        selected_following, selected_follower = _sample_directional_neighbors(
            node_id,
            texts,
            context,
            following_quota=following_quota,
            follower_quota=follower_quota,
        )
        following_texts = [_node_text(texts[n]) for n in selected_following]
        follower_texts = [_node_text(texts[n]) for n in selected_follower]
        ego_prompt, one_hop_prompt = _make_relation_aware_prompts(ego_text, following_texts, follower_texts, classes)
        prompts["ego"].append(ego_prompt)
        prompts["directional_1hop"].append(one_hop_prompt)
        counts["ego_nodes"].append(1)
        counts["following_nodes"].append(len(following_texts))
        counts["follower_nodes"].append(len(follower_texts))
        counts["directional_1hop_nodes"].append(1 + len(following_texts) + len(follower_texts))
    return {
        "prompt_family": "relation_aware",
        "prompt_style": "instruction_query_directional_social_context",
        "prompt_components": prompts,
        "component_max_length_group": {"ego": "ego", "directional_1hop": "hop"},
        "counts": counts,
        "neighbor_sample_policy": "directional_heuristic_fixed_quota",
        "directional_quota": {"following": int(following_quota), "follower": int(follower_quota)},
        "component_order": {
            "relation_aware_ego": ["ego"],
            "relation_aware_1hop": ["directional_1hop"],
        },
    }


def _expert_output_name(prompt_mode, prompt_family_version="v1", embedding_encoder_tag="qwen3"):
    if str(prompt_family_version).lower() == "v2":
        names = {
            "expert_ego": f"glance_prompt_expert_ego_{embedding_encoder_tag}_embed.pt",
            "expert_graph_following": f"glance_prompt_expert_graph_following_{embedding_encoder_tag}_embed.pt",
            "expert_graph_follower": f"glance_prompt_expert_graph_follower_{embedding_encoder_tag}_embed.pt",
            "expert_tweet": f"glance_prompt_expert_tweet_{embedding_encoder_tag}_embed.pt",
            "expert_conflict": f"glance_prompt_expert_conflict_{embedding_encoder_tag}_embed.pt",
            "expert_concat_v1": f"glance_prompt_expert_concat_v2_{embedding_encoder_tag}_embed.pt",
        }
    else:
        names = {
            "expert_ego": "glance_prompt_expert_ego_qwen3_embed.pt",
            "expert_graph_following": "glance_prompt_expert_graph_following_qwen3_embed.pt",
            "expert_graph_follower": "glance_prompt_expert_graph_follower_qwen3_embed.pt",
            "expert_tweet": "glance_prompt_expert_tweet_qwen3_embed.pt",
            "expert_conflict": "glance_prompt_expert_conflict_qwen3_embed.pt",
            "expert_concat_v1": "glance_prompt_expert_concat_v1_qwen3_embed.pt",
        }
    return names[prompt_mode]


def _default_output_path(dataset_path: Path, prompt_mode: str, prompt_family_version="v1", embedding_encoder_tag="qwen3"):
    if prompt_mode == "glance_concat_ego_hop1_hop2":
        return dataset_path / DEFAULT_OUTPUT_NAME
    if prompt_mode in EXPERT_PROMPT_MODES:
        return dataset_path / _expert_output_name(
            prompt_mode,
            prompt_family_version=prompt_family_version,
            embedding_encoder_tag=embedding_encoder_tag,
        )
    return dataset_path / f"glance_qwen3_prompt_cache_{prompt_mode}.pt"


def _graph_prompt(
    direction_name,
    ego_record,
    neighbor_records,
    total_count,
    reciprocal_count,
):
    heading = "FOLLOWING_NEIGHBORS" if direction_name == "following" else "FOLLOWER_NEIGHBORS"
    if direction_name == "following":
        instruct = (
            "Instruct: Encode who this account chooses to follow and what that implies about "
            "social role, coordination, fandom, promotion, or organic behavior."
        )
    else:
        instruct = (
            "Instruct: Encode who follows this account and what that implies about audience type, "
            "credibility, coordination, or suspicious amplification."
        )
    neighbor_lines = [_neighbor_card(record) for record in neighbor_records] or ["- None"]
    social_hints = [
        f"count_{direction_name}: {int(total_count)}",
        f"has_{direction_name}: {_format_bool(total_count > 0)}",
        f"reciprocal_{direction_name}_count: {int(reciprocal_count)}",
    ]
    query = "\n".join(
        [
            "Query:",
            "EGO_PROFILE_BRIEF:",
            _profile_card(ego_record, brief=True),
            f"{heading}:",
            *neighbor_lines,
            "SOCIAL_HINTS:",
            *social_hints,
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _graph_prompt_partitioned(
    direction_name,
    ego_record,
    support_records,
    contrast_records,
    total_count,
    reciprocal_count,
    candidate_count,
    selected_count,
    mean_sim_support,
    mean_sim_contrast,
    reciprocal_ratio_selected,
):
    heading_prefix = "FOLLOWING" if direction_name == "following" else "FOLLOWER"
    if direction_name == "following":
        instruct = (
            "Instruct: Encode who this account chooses to follow, separating supportive neighbors from contrasting "
            "neighbors to capture social role, coordination, fandom, promotion, or organic behavior."
        )
    else:
        instruct = (
            "Instruct: Encode who follows this account, separating supportive neighbors from contrasting neighbors "
            "to capture audience type, credibility, coordination, or suspicious amplification."
        )
    support_lines = [_neighbor_card(record) for record in support_records] or ["- None"]
    contrast_lines = [_neighbor_card(record) for record in contrast_records] or ["- None"]
    social_hints = [
        f"count_{direction_name}: {int(total_count)}",
        f"candidate_count_{direction_name}: {int(candidate_count)}",
        f"selected_count_{direction_name}: {int(selected_count)}",
        f"has_{direction_name}: {_format_bool(total_count > 0)}",
        f"reciprocal_{direction_name}_count: {int(reciprocal_count)}",
        f"mean_sim_{direction_name}_support: {float(mean_sim_support):.4f}",
        f"mean_sim_{direction_name}_contrast: {float(mean_sim_contrast):.4f}",
        f"reciprocal_ratio_{direction_name}_selected: {float(reciprocal_ratio_selected):.4f}",
    ]
    query = "\n".join(
        [
            "Query:",
            "EGO_PROFILE_BRIEF:",
            _profile_card(ego_record, brief=True),
            f"{heading_prefix}_SUPPORT:",
            *support_lines,
            f"{heading_prefix}_CONTRAST:",
            *contrast_lines,
            "SOCIAL_HINTS:",
            *social_hints,
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _tweet_prompt(record, tweet_stats):
    instruct = "Instruct: Encode the posting behavior of the account for downstream bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_BRIEF:",
            _profile_card(record, brief=True),
            "TWEET_BEHAVIOR_SUMMARY:",
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
            "Focus on topical consistency, conversationality, promotion intensity, repetition, and automation cues.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _conflict_flags(record, tweet_stats, count_following, count_follower):
    sparse_profile = int((not record["bio_present"]) and (not record["location"]) and record["listed_count"] == 0)
    retweet_dominant = int(tweet_stats["rt_ratio"] >= 0.6)
    promo_heavy = int(max(tweet_stats["url_ratio"], tweet_stats["hashtag_ratio"]) >= 0.5)
    directional_sparse = int((int(count_following) + int(count_follower)) <= 1)
    return {
        "sparse_profile": sparse_profile,
        "retweet_dominant": retweet_dominant,
        "promo_heavy": promo_heavy,
        "directional_sparse": directional_sparse,
    }


def _conflict_mismatch_hints(flags, tweet_stats, count_following, count_follower, selection_summary=None):
    mismatch_hints = []
    if flags["sparse_profile"] and tweet_stats["tweet_count"] > 0:
        mismatch_hints.append("- Sparse profile but active tweeting behavior.")
    if flags["retweet_dominant"]:
        mismatch_hints.append("- Posting behavior is dominated by retweets.")
    if flags["promo_heavy"]:
        mismatch_hints.append("- Posting behavior contains heavy promotion or link-sharing.")
    if flags["directional_sparse"]:
        mismatch_hints.append("- Social neighborhood is very sparse in the directed graph.")
    if int(count_following) > 0 and int(count_follower) == 0:
        mismatch_hints.append("- Follows others but does not attract follower context.")
    if int(count_follower) > 0 and int(count_following) == 0:
        mismatch_hints.append("- Attracts followers without meaningful following context.")
    if selection_summary is not None:
        if float(selection_summary.get("mean_sim_following_contrast", 0.0)) < float(selection_summary.get("mean_sim_following_support", 0.0)):
            mismatch_hints.append("- Following neighborhood contains a support-versus-contrast semantic split.")
        if float(selection_summary.get("mean_sim_follower_contrast", 0.0)) < float(selection_summary.get("mean_sim_follower_support", 0.0)):
            mismatch_hints.append("- Follower neighborhood contains a support-versus-contrast semantic split.")
    return mismatch_hints


def _conflict_prompt(record, tweet_stats, following_records, follower_records, count_following, count_follower):
    flags = _conflict_flags(record, tweet_stats, count_following, count_follower)
    mismatch_hints = _conflict_mismatch_hints(flags, tweet_stats, count_following, count_follower)
    following_lines = [_neighbor_card(item) for item in following_records] or ["- None"]
    follower_lines = [_neighbor_card(item) for item in follower_records] or ["- None"]
    instruct = "Instruct: Encode cross-view consistency and inconsistency cues for bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_CARD:",
            _profile_card(record, brief=False),
            "TWEET_CARD:",
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
            "GRAPH_CARD_FOLLOWING:",
            *following_lines,
            "GRAPH_CARD_FOLLOWER:",
            *follower_lines,
            "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
            "Focus on whether the profile, posting behavior, and social neighborhood support or contradict each other.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}", flags


def _conflict_prompt_partitioned(
    record,
    tweet_stats,
    following_support_records,
    following_contrast_records,
    follower_support_records,
    follower_contrast_records,
    count_following,
    count_follower,
    selection_summary,
):
    flags = _conflict_flags(record, tweet_stats, count_following, count_follower)
    mismatch_hints = _conflict_mismatch_hints(
        flags,
        tweet_stats,
        count_following,
        count_follower,
        selection_summary=selection_summary,
    )
    following_support_lines = [_neighbor_card(item) for item in following_support_records] or ["- None"]
    following_contrast_lines = [_neighbor_card(item) for item in following_contrast_records] or ["- None"]
    follower_support_lines = [_neighbor_card(item) for item in follower_support_records] or ["- None"]
    follower_contrast_lines = [_neighbor_card(item) for item in follower_contrast_records] or ["- None"]
    instruct = "Instruct: Encode cross-view consistency and inconsistency cues for bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_CARD:",
            _profile_card(record, brief=False),
            "TWEET_CARD:",
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
            "FOLLOWING_SUPPORT:",
            *following_support_lines,
            "FOLLOWING_CONTRAST:",
            *following_contrast_lines,
            "FOLLOWER_SUPPORT:",
            *follower_support_lines,
            "FOLLOWER_CONTRAST:",
            *follower_contrast_lines,
            "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
            "Focus on whether the profile, posting behavior, and relation-aware support/contrast neighborhoods support or contradict each other.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}", flags


def _ego_explain_generation_prompt(record):
    system = _summary_generation_system()
    user = "\n".join(
        [
            "Summarize the profile card using identity presentation, reach/activity cues, and any missing or unusual profile fields.",
            *_summary_prompt_rules(),
            "PROFILE_CARD:",
            _profile_card(record, brief=False),
        ]
    )
    return {"system": system, "user": user}


def _ego_embedding_prompt(explanation_text):
    return _expert_embedding_prompt("ego", explanation_text)


def _resolve_expert_prompt_bundle(args, records, edge_index, edge_type, target_node_ids=None, selection_features=None):
    num_nodes = len(records)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    mode = str(args.prompt_mode)
    prompt_family_version = str(getattr(args, "prompt_family_version", "v1")).lower()
    policy = str(args.neighbor_sampling_policy).lower()
    if prompt_family_version not in PROMPT_FAMILY_VERSION_CHOICES:
        raise ValueError(f"Unsupported prompt_family_version: {prompt_family_version}")
    if prompt_family_version == "v2" and policy == "center_induced_relation_aware":
        raise ValueError("prompt_expert_bundle_v2 keeps direction-split ranked neighbors as the mainline and does not expose center_induced_relation_aware.")
    selected_component_map_v1 = {
        "expert_ego": ["ego"],
        "expert_graph_following": ["graph_following"],
        "expert_graph_follower": ["graph_follower"],
        "expert_tweet": ["tweet"],
        "expert_conflict": ["conflict"],
        "expert_concat_v1": list(EXPERT_COMPONENT_NAMES_WITH_STRUCTURED),
    }
    selected_component_map_v2 = {
        "expert_ego": ["ego"],
        "expert_graph_following": ["graph_following"],
        "expert_graph_follower": ["graph_follower"],
        "expert_tweet": ["tweet"],
        "expert_conflict": ["conflict"],
        "expert_concat_v1": list(EXPERT_COMPONENT_NAMES_V2_WITH_STRUCTURED),
    }
    selected_components = (
        selected_component_map_v2 if prompt_family_version == "v2" else selected_component_map_v1
    )[mode]
    prompt_components = {
        name: []
        for name in selected_components
        if not (prompt_family_version == "v1" and name == "ego")
    }
    component_prompt_roles = {
        "ego": "profile_explainer",
        "graph_following": "following_role_explainer",
        "graph_follower": "follower_audience_explainer",
        "tweet": "posting_behavior_explainer",
        "conflict": "cross_view_conflict_explainer",
        "metadata_structured": "structured_profile_metadata_encoder",
    }
    counts = {
        "ego_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "tweet_nodes": [],
        "conflict_nodes": [],
    }
    scalar_features = {key: [] for key in EXPERT_SCALAR_KEYS}
    structured_components = {"metadata_structured": []}
    prompt_rows = []
    generation_rows_by_component = {}
    if prompt_family_version == "v2":
        generation_components = set(selected_components)
        if "conflict" in selected_components:
            generation_components.update({"tweet", "graph_following", "graph_follower"})
        generation_rows_by_component = {name: [] for name in generation_components}
    elif "ego" in selected_components:
        generation_rows_by_component = {"ego": []}

    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    for node_id in target_ids:
        record = records[node_id]
        if policy == "center_induced_relation_aware":
            if selection_features is None:
                raise ValueError("center_induced_relation_aware expert prompts require selection_features.")
            center_selection = _select_center_induced_directional_neighbors(
                node_id,
                [item["raw_text"] for item in records],
                context,
                selection_features,
                following_quota=int(args.following_quota),
                follower_quota=int(args.follower_quota),
            )
            following_support_stats = center_selection["following_support"]
            following_contrast_stats = center_selection["following_contrast"]
            follower_support_stats = center_selection["follower_support"]
            follower_contrast_stats = center_selection["follower_contrast"]
            following_ids = [int(item["candidate"]) for item in (following_support_stats + following_contrast_stats)]
            follower_ids = [int(item["candidate"]) for item in (follower_support_stats + follower_contrast_stats)]
            selection_summary = center_selection["summary"]
        else:
            following_candidates = _rank_directional_candidates(node_id, context["following"][node_id], [item["raw_text"] for item in records], context)
            follower_candidates = _rank_directional_candidates(node_id, context["follower"][node_id], [item["raw_text"] for item in records], context)
            following_ids = following_candidates[: int(args.following_quota)]
            follower_ids = follower_candidates[: int(args.follower_quota)]
            following_support_stats = []
            following_contrast_stats = []
            follower_support_stats = []
            follower_contrast_stats = []
            selection_summary = {
                "candidate_count_following": int(len(set(context["following"][node_id]))),
                "candidate_count_follower": int(len(set(context["follower"][node_id]))),
                "selected_count_following": int(len(following_ids)),
                "selected_count_follower": int(len(follower_ids)),
                "mean_sim_following_support": 0.0,
                "mean_sim_following_contrast": 0.0,
                "mean_sim_follower_support": 0.0,
                "mean_sim_follower_contrast": 0.0,
                "reciprocal_ratio_following_selected": 0.0,
                "reciprocal_ratio_follower_selected": 0.0,
                "fallback_used": False,
            }
        following_records = [records[idx] for idx in following_ids]
        follower_records = [records[idx] for idx in follower_ids]
        following_support_records = [records[int(item["candidate"])] for item in following_support_stats]
        following_contrast_records = [records[int(item["candidate"])] for item in following_contrast_stats]
        follower_support_records = [records[int(item["candidate"])] for item in follower_support_stats]
        follower_contrast_records = [records[int(item["candidate"])] for item in follower_contrast_stats]
        reciprocal_count = len(set(context["following"][node_id]).intersection(set(context["follower"][node_id])))
        tweet_stats = _get_tweet_stats(record, sample_size=int(getattr(args, "tweet_sample_size", 6)))
        count_following = len(set(context["following"][node_id]))
        count_follower = len(set(context["follower"][node_id]))
        structured_components["metadata_structured"].append(_metadata_structured_features(record))
        scalar_features["count_following"].append(float(len(set(context["following"][node_id]))))
        scalar_features["count_follower"].append(float(len(set(context["follower"][node_id]))))
        scalar_features["has_following"].append(float(1 if context["following"][node_id] else 0))
        scalar_features["has_follower"].append(float(1 if context["follower"][node_id] else 0))
        scalar_features["candidate_count_following"].append(float(selection_summary["candidate_count_following"]))
        scalar_features["candidate_count_follower"].append(float(selection_summary["candidate_count_follower"]))
        scalar_features["selected_count_following"].append(float(selection_summary["selected_count_following"]))
        scalar_features["selected_count_follower"].append(float(selection_summary["selected_count_follower"]))
        scalar_features["mean_sim_following_support"].append(float(selection_summary["mean_sim_following_support"]))
        scalar_features["mean_sim_following_contrast"].append(float(selection_summary["mean_sim_following_contrast"]))
        scalar_features["mean_sim_follower_support"].append(float(selection_summary["mean_sim_follower_support"]))
        scalar_features["mean_sim_follower_contrast"].append(float(selection_summary["mean_sim_follower_contrast"]))
        scalar_features["reciprocal_ratio_following_selected"].append(float(selection_summary["reciprocal_ratio_following_selected"]))
        scalar_features["reciprocal_ratio_follower_selected"].append(float(selection_summary["reciprocal_ratio_follower_selected"]))
        scalar_features["rt_ratio"].append(float(tweet_stats["rt_ratio"]))
        scalar_features["url_ratio"].append(float(tweet_stats["url_ratio"]))
        scalar_features["hashtag_ratio"].append(float(tweet_stats["hashtag_ratio"]))
        counts["ego_nodes"].append(1)
        counts["following_nodes"].append(len(following_records))
        counts["follower_nodes"].append(len(follower_records))
        counts["tweet_nodes"].append(len(tweet_stats["sampled_tweets"]))
        counts["conflict_nodes"].append(
            1 + len(following_records) + len(follower_records) + len(tweet_stats["sampled_tweets"])
        )
        conflict_flags = _conflict_flags(record, tweet_stats, count_following, count_follower)
        conflict_mismatch_hints = _conflict_mismatch_hints(
            conflict_flags,
            tweet_stats,
            count_following,
            count_follower,
            selection_summary=selection_summary if policy == "center_induced_relation_aware" else None,
        )

        row = {
            "node_id": int(node_id),
            "selected_components": list(selected_components),
            "profile_brief": _profile_card(record, brief=True),
            "profile_cue_summary": _profile_cue_summary(record, include_identity=True),
            "tweet_summary": _tweet_behavior_summary(record, tweet_stats),
            "tweet_source": str(record.get("tweet_source", "norm_user_text")),
            "raw_tweet_count": int(record.get("raw_tweet_count", len(record.get("tweets", [])))),
            "sampled_tweet_count": int(len(tweet_stats.get("sampled_tweets", []))),
            "sampled_tweet_ids": list(record.get("raw_tweet_ids", []))[: int(len(tweet_stats.get("sampled_tweets", [])))],
            "tweet_sampling_profile": dict(tweet_stats.get("sampling_profile", {})) if isinstance(tweet_stats.get("sampling_profile"), dict) else {},
            "neighbor_selection_policy": policy,
            "candidate_count_following": int(selection_summary["candidate_count_following"]),
            "candidate_count_follower": int(selection_summary["candidate_count_follower"]),
            "selected_count_following": int(selection_summary["selected_count_following"]),
            "selected_count_follower": int(selection_summary["selected_count_follower"]),
            "mean_sim_following_support": float(selection_summary["mean_sim_following_support"]),
            "mean_sim_following_contrast": float(selection_summary["mean_sim_following_contrast"]),
            "mean_sim_follower_support": float(selection_summary["mean_sim_follower_support"]),
            "mean_sim_follower_contrast": float(selection_summary["mean_sim_follower_contrast"]),
            "reciprocal_ratio_following_selected": float(selection_summary["reciprocal_ratio_following_selected"]),
            "reciprocal_ratio_follower_selected": float(selection_summary["reciprocal_ratio_follower_selected"]),
            "selection_fallback_used": bool(selection_summary["fallback_used"]),
            "following_support_ids": [int(item["candidate"]) for item in following_support_stats],
            "following_contrast_ids": [int(item["candidate"]) for item in following_contrast_stats],
            "follower_support_ids": [int(item["candidate"]) for item in follower_support_stats],
            "follower_contrast_ids": [int(item["candidate"]) for item in follower_contrast_stats],
            "conflict_flags": conflict_flags,
            "conflict_mismatch_hints": conflict_mismatch_hints,
            "component_prompt_roles": component_prompt_roles,
        }

        if "ego" in generation_rows_by_component:
            generation_prompt = _ego_explain_generation_prompt(record)
            generation_rows_by_component["ego"].append(
                {
                    "node_id": int(node_id),
                    "component_name": "ego",
                    "system": generation_prompt["system"],
                    "user": generation_prompt["user"],
                    "profile_card": _profile_card(record, brief=False),
                    "fallback_explanation": _deterministic_ego_explain_fallback(record),
                    "prompt_role": component_prompt_roles["ego"],
                }
            )
            row["ego_generation_prompt"] = generation_prompt["user"]

        if prompt_family_version == "v2":
            following_summary = {
                "count": count_following,
                "selected_count": int(selection_summary["selected_count_following"]),
                "reciprocal_ratio_selected": float(selection_summary["reciprocal_ratio_following_selected"]),
            }
            follower_summary = {
                "count": count_follower,
                "selected_count": int(selection_summary["selected_count_follower"]),
                "reciprocal_ratio_selected": float(selection_summary["reciprocal_ratio_follower_selected"]),
            }
            if "graph_following" in generation_rows_by_component:
                generation_prompt = _graph_explain_generation_prompt(
                    "following",
                    record,
                    following_records,
                    following_summary,
                )
                generation_rows_by_component["graph_following"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "graph_following",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": component_prompt_roles["graph_following"],
                    }
                )
                row["graph_following_generation_prompt"] = generation_prompt["user"]
            if "graph_follower" in generation_rows_by_component:
                generation_prompt = _graph_explain_generation_prompt(
                    "follower",
                    record,
                    follower_records,
                    follower_summary,
                )
                generation_rows_by_component["graph_follower"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "graph_follower",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": component_prompt_roles["graph_follower"],
                    }
                )
                row["graph_follower_generation_prompt"] = generation_prompt["user"]
            if "tweet" in generation_rows_by_component:
                generation_prompt = _tweet_explain_generation_prompt(record, tweet_stats)
                generation_rows_by_component["tweet"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "tweet",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": component_prompt_roles["tweet"],
                    }
                )
                row["tweet_generation_prompt"] = generation_prompt["user"]
        else:
            if "graph_following" in selected_components:
                if policy == "center_induced_relation_aware":
                    prompt_components["graph_following"].append(
                        _graph_prompt_partitioned(
                            "following",
                            record,
                            following_support_records,
                            following_contrast_records,
                            total_count=count_following,
                            reciprocal_count=reciprocal_count,
                            candidate_count=selection_summary["candidate_count_following"],
                            selected_count=selection_summary["selected_count_following"],
                            mean_sim_support=selection_summary["mean_sim_following_support"],
                            mean_sim_contrast=selection_summary["mean_sim_following_contrast"],
                            reciprocal_ratio_selected=selection_summary["reciprocal_ratio_following_selected"],
                        )
                    )
                else:
                    prompt_components["graph_following"].append(
                        _graph_prompt(
                            "following",
                            record,
                            following_records,
                            total_count=count_following,
                            reciprocal_count=reciprocal_count,
                        )
                    )
                row["graph_following_prompt"] = prompt_components["graph_following"][-1]

            if "graph_follower" in selected_components:
                if policy == "center_induced_relation_aware":
                    prompt_components["graph_follower"].append(
                        _graph_prompt_partitioned(
                            "follower",
                            record,
                            follower_support_records,
                            follower_contrast_records,
                            total_count=count_follower,
                            reciprocal_count=reciprocal_count,
                            candidate_count=selection_summary["candidate_count_follower"],
                            selected_count=selection_summary["selected_count_follower"],
                            mean_sim_support=selection_summary["mean_sim_follower_support"],
                            mean_sim_contrast=selection_summary["mean_sim_follower_contrast"],
                            reciprocal_ratio_selected=selection_summary["reciprocal_ratio_follower_selected"],
                        )
                    )
                else:
                    prompt_components["graph_follower"].append(
                        _graph_prompt(
                            "follower",
                            record,
                            follower_records,
                            total_count=count_follower,
                            reciprocal_count=reciprocal_count,
                        )
                    )
                row["graph_follower_prompt"] = prompt_components["graph_follower"][-1]

            if "tweet" in selected_components:
                prompt_components["tweet"].append(_tweet_prompt(record, tweet_stats))
                row["tweet_prompt"] = prompt_components["tweet"][-1]

        if prompt_family_version == "v1" and "conflict" in selected_components:
            if policy == "center_induced_relation_aware":
                conflict_prompt, flags = _conflict_prompt_partitioned(
                    record,
                    tweet_stats,
                    following_support_records,
                    following_contrast_records,
                    follower_support_records,
                    follower_contrast_records,
                    count_following=count_following,
                    count_follower=count_follower,
                    selection_summary=selection_summary,
                )
            else:
                conflict_prompt, flags = _conflict_prompt(
                    record,
                    tweet_stats,
                    following_records,
                    follower_records,
                    count_following=count_following,
                    count_follower=count_follower,
                )
            prompt_components["conflict"].append(conflict_prompt)
            row["conflict_prompt"] = conflict_prompt
            row["conflict_flags"] = flags

        prompt_rows.append(row)

    if prompt_family_version == "v2":
        prompt_family = "prompt_expert_bundle_v2"
        prompt_style = "expert_prompt_bundle_v2_explanation_first"
        semantic_view_mode = "prompt_expert_bundle_v2"
        selection_policy_note = (
            "Direction-split fixed-quota ranking keeps follower and following separate; "
            "neighbors are ordered by text presence, reciprocity, shared-neighbor activity, structural activity, and usable text length."
        )
    else:
        prompt_family = "prompt_expert_bundle_center_induced_v1" if policy == "center_induced_relation_aware" else "prompt_expert_bundle_v1"
        prompt_style = "expert_prompt_bundle_center_induced_v1" if policy == "center_induced_relation_aware" else "expert_prompt_bundle_v1"
        semantic_view_mode = "prompt_expert_bundle_center_induced_v1" if policy == "center_induced_relation_aware" else "prompt_expert_bundle_v1"
        selection_policy_note = (
            "Support neighbors are ranked by high cosine similarity plus reciprocity/common-neighbor tie-breakers; "
            "contrast neighbors are ranked by low cosine similarity with non-trivial structure and text availability."
            if policy == "center_induced_relation_aware"
            else "Directional heuristic fixed-quota ranking."
        )

    return {
        "prompt_family": prompt_family,
        "prompt_style": prompt_style,
        "prompt_family_version": prompt_family_version,
        "prompt_components": prompt_components,
        "structured_components": structured_components,
        "structured_component_schema": {
            "metadata_structured": list(METADATA_STRUCTURED_FEATURE_NAMES),
        },
        "generation_rows_by_component": generation_rows_by_component,
        "component_max_length_group": {
            "ego": "ego",
            "graph_following": "hop",
            "graph_follower": "hop",
            "tweet": "hop",
            "conflict": "hop",
        },
        "component_prompt_roles": component_prompt_roles,
        "counts": counts,
        "neighbor_sample_policy": "center_induced_relation_aware" if policy == "center_induced_relation_aware" else "directional_heuristic_fixed_quota",
        "directional_quota": {"following": int(args.following_quota), "follower": int(args.follower_quota)},
        "support_contrast_quota": {
            "following": {"support": int(math.ceil(float(args.following_quota) / 2.0)), "contrast": int(math.floor(float(args.following_quota) / 2.0))},
            "follower": {"support": int(math.ceil(float(args.follower_quota) / 2.0)), "contrast": int(math.floor(float(args.follower_quota) / 2.0))},
        },
        "component_order": {mode_name: list(component_names) for mode_name, component_names in {
            "expert_ego": ["ego"],
            "expert_graph_following": ["graph_following"],
            "expert_graph_follower": ["graph_follower"],
            "expert_tweet": ["tweet"],
            "expert_conflict": ["conflict"],
            "expert_concat_v1": (
                list(EXPERT_COMPONENT_NAMES_V2_WITH_STRUCTURED)
                if prompt_family_version == "v2"
                else list(EXPERT_COMPONENT_NAMES_WITH_STRUCTURED)
            ),
        }.items()},
        "selected_components": list(selected_components),
        "scalar_features": scalar_features,
        "prompt_rows": prompt_rows,
        "semantic_view_mode": semantic_view_mode,
        "selection_policy_note": selection_policy_note,
        "needs_conflict_generation": bool(prompt_family_version == "v2" and "conflict" in selected_components),
    }


def _resolve_prompt_bundle(args, texts, edge_index, edge_type, classes, target_node_ids=None, selection_features=None):
    mode = str(args.prompt_mode)
    policy = str(args.neighbor_sampling_policy).lower()
    if mode.startswith("glance_"):
        if policy not in {"auto", "uniform_random"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'uniform_random' or 'auto'.")
        bundle = _build_glance_prompt_bundle(
            texts,
            edge_index,
            edge_type,
            cap=int(args.neighbor_cap),
            seed=int(args.seed),
            classes=classes,
            target_node_ids=target_node_ids,
        )
    elif mode.startswith("relation_aware_"):
        if policy not in {"auto", "directional_heuristic"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'directional_heuristic' or 'auto'.")
        bundle = _build_relation_aware_prompt_bundle(
            texts,
            edge_index,
            edge_type,
            following_quota=int(args.following_quota),
            follower_quota=int(args.follower_quota),
            classes=classes,
            target_node_ids=target_node_ids,
        )
    elif mode.startswith("expert_"):
        if policy not in {"auto", "directional_heuristic", "center_induced_relation_aware"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'directional_heuristic', 'center_induced_relation_aware', or 'auto'.")
        records = [_parse_norm_user_record(raw_text, idx) for idx, raw_text in enumerate(texts)]
        bundle = _resolve_expert_prompt_bundle(
            args,
            records,
            edge_index,
            edge_type,
            target_node_ids=target_node_ids,
            selection_features=selection_features,
        )
    else:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    selected_components = bundle["component_order"].get(mode)
    if not selected_components:
        raise ValueError(f"Prompt bundle did not define selected components for mode {mode}.")
    bundle["prompt_mode"] = mode
    bundle["selected_components"] = list(selected_components)
    bundle["neighbor_sampling_policy_requested"] = policy
    return bundle


def _selected_components_require_tweet(prompt_bundle):
    selected_components = set(prompt_bundle.get("selected_components", []))
    if "tweet" in selected_components:
        return True
    if "conflict" in selected_components:
        return True
    return bool(prompt_bundle.get("needs_conflict_generation"))


def _device_from_args(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _effective_explain_batch_size(args, device):
    requested = int(getattr(args, "explain_batch_size", 0) or 0)
    if requested > 0:
        return requested
    return 2 if getattr(device, "type", "cpu") == "cuda" else 1


def _load_embedding_model(args, device):
    from transformers import AutoModel, AutoTokenizer

    requested_encoder_tag = _infer_embedding_encoder_tag(args.model_path)
    checkpoint_path = _resolve_finetuned_roberta_checkpoint(args) if requested_encoder_tag == "roberta_finetuned" else None
    model_source = _require_local_pretrained_source(args.model_path, model_role="Embedding model")
    encoder_tag = requested_encoder_tag if requested_encoder_tag == "roberta_finetuned" else _infer_embedding_encoder_tag(model_source)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
        torch_dtype=torch.float32 if encoder_tag == "roberta_finetuned" else (torch.float16 if device.type == "cuda" else torch.float32),
        # RoBERTa-style encoders are small enough to load eagerly, and forcing
        # low_cpu_mem_usage here can leave meta tensors that fail on .to(device).
        low_cpu_mem_usage=False,
    ).to(device)
    checkpoint_load_summary = None
    if checkpoint_path is not None:
        checkpoint_load_summary = _load_simteg_lm_checkpoint_into_encoder(model, checkpoint_path)
        checkpoint_load_summary["checkpoint_path"] = str(checkpoint_path)
        model.to(device)
    model.eval()
    pooling_mode = _resolve_pooling_mode(model_source, model)
    # Qwen embedding cards use left padding plus last-token pooling; mean-pool encoders stay right-padded.
    tokenizer.padding_side = "left" if pooling_mode == "last_token" else "right"
    return tokenizer, model, model_source, pooling_mode, checkpoint_load_summary


def _encode_texts(model, tokenizer, texts, device, batch_size, max_length, normalize, pooling_mode):
    chunks = []
    effective_max_length = _effective_embedding_max_length(tokenizer, model, max_length)
    if pooling_mode == "simteg_mean":
        effective_max_length = min(int(effective_max_length), 512)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                add_special_tokens=False if pooling_mode == "simteg_mean" else True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=(pooling_mode == "simteg_mean"))
            if pooling_mode == "last_token":
                pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            elif pooling_mode == "masked_mean":
                pooled = masked_mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            elif pooling_mode == "simteg_mean":
                pooled = outputs.hidden_states[-1].mean(dim=1)
            else:
                raise ValueError(f"Unsupported embedding pooling mode: {pooling_mode}")
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=torch.float32)


def _component_effective_token_budget(tokenizer, model, requested_budget, pooling_mode=None):
    effective = int(_effective_embedding_max_length(tokenizer, model, int(requested_budget)))
    if pooling_mode == "simteg_mean":
        effective = min(effective, 512)
    return effective


def _format_chat_prompt(tokenizer, system_prompt, user_prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def _load_generation_model(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.explain_model_path:
        raise ValueError(
            f"Prompt mode {args.prompt_mode} requires --explain_model_path for explanation generation."
        )
    model_source = _require_local_pretrained_source(args.explain_model_path, model_role="Explain model")
    model_source_text = str(model_source).lower()
    is_qwen_generation_model = "qwen" in model_source_text
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(args.explain_trust_remote_code),
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_dtype = torch.float32
    if device.type == "cuda":
        # Qwen2.5-Instruct generation is numerically unstable on this stack when
        # forced to fp16, producing punctuation-only continuations. Let HF honor
        # the checkpoint dtype for Qwen; keep the old fp16 path for smaller
        # non-Qwen local explainers.
        generation_dtype = "auto" if is_qwen_generation_model else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=bool(args.explain_trust_remote_code),
        local_files_only=True,
        torch_dtype=generation_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def _resolve_generation_runtime(args, device):
    if not args.explain_model_path:
        return None
    try:
        tokenizer, model = _load_generation_model(args, device)
    except Exception as exc:
        if bool(getattr(args, "explain_required", False)):
            raise RuntimeError(
                "Failed to load the required local explain model. "
                "Pass a valid local HuggingFace causal-LM snapshot via --explain_model_path."
            ) from exc
        return {"load_error": str(exc)}
    return {"tokenizer": tokenizer, "model": model}


def _generate_component_explanations(
    args,
    generation_rows,
    device,
    generation_runtime=None,
    component_name=None,
    sidecar_path=None,
    wandb_run=None,
    wandb_step_base=0,
):
    component_name = str(component_name or (generation_rows[0].get("component_name", "unknown") if generation_rows else "unknown"))
    quality_gate = bool(getattr(args, "explain_quality_gate", True))
    if not generation_rows:
        return [], {
            "generated_count": 0,
            "fallback_count": 0,
            "resumed_count": 0,
            "requested_count": 0,
            "total_count": 0,
            "generation_mode": "skipped_empty",
            "explain_model_path": str(args.explain_model_path or ""),
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
            "explain_model_load_mode": "skipped_empty",
            "explanation_sidecar_path": str(sidecar_path or ""),
            "explain_quality_gate": bool(quality_gate),
            "resume_rejected_count": 0,
            "resume_rejected_reasons": {},
            "quality_rejected_count": 0,
            "quality_rejected_reasons": {},
        }
    prompts = [
        _format_chat_prompt(generation_runtime.get("tokenizer"), row["system"], row["user"])
        if generation_runtime is not None and generation_runtime.get("tokenizer") is not None
        else None
        for row in generation_rows
    ]
    prompt_hashes = [
        _string_sha256(prompt if prompt is not None else f"{row.get('system', '')}\n{row.get('user', '')}")
        for prompt, row in zip(prompts, generation_rows)
    ]
    if sidecar_path:
        resumed_by_key, resume_rejected_reasons = _load_resume_explanations(
            sidecar_path,
            component_name,
            quality_gate=quality_gate,
        )
    else:
        resumed_by_key, resume_rejected_reasons = {}, {}
    resume_rejected_count = int(sum(int(value) for value in resume_rejected_reasons.values()))
    explanations = [None for _ in generation_rows]
    resumed_count = 0
    pending_rows = []
    pending_hashes = []
    pending_positions = []
    for idx, (row, prompt_hash) in enumerate(zip(generation_rows, prompt_hashes)):
        key = (int(row["node_id"]), str(prompt_hash))
        if key in resumed_by_key:
            explanations[idx] = resumed_by_key[key]
            resumed_count += 1
        else:
            pending_rows.append(row)
            pending_hashes.append(prompt_hash)
            pending_positions.append(idx)
    if resumed_count:
        print(
            {
                "status": "resume_explanations",
                "component": component_name,
                "resumed_count": int(resumed_count),
                "pending_count": int(len(pending_rows)),
                "resume_rejected_count": int(resume_rejected_count),
                "resume_rejected_reasons": dict(resume_rejected_reasons),
                "sidecar_path": str(sidecar_path or ""),
            }
        )
    elif resume_rejected_count:
        print(
            {
                "status": "resume_explanations_quality_rejected",
                "component": component_name,
                "resume_rejected_count": int(resume_rejected_count),
                "resume_rejected_reasons": dict(resume_rejected_reasons),
                "pending_count": int(len(pending_rows)),
                "sidecar_path": str(sidecar_path or ""),
            }
        )
    if not args.explain_model_path:
        if bool(getattr(args, "explain_required", False)):
            raise ValueError(
                "--explain_required was set but --explain_model_path is empty. "
                "Provide a local instruct/causal-LM snapshot path for real explanation generation."
            )
        pending_explanations = _deterministic_explanations(pending_rows)
        for position, row, prompt_hash, explanation in zip(pending_positions, pending_rows, pending_hashes, pending_explanations):
            explanations[position] = explanation
            if sidecar_path:
                _append_jsonl(
                    sidecar_path,
                    [
                        {
                            "node_id": int(row["node_id"]),
                            "component_name": component_name,
                            "prompt_role": row.get("prompt_role", ""),
                            "prompt_hash": prompt_hash,
                            "explanation": explanation,
                            "generation_mode": "deterministic_fallback",
                        }
                    ],
                )
        return explanations, {
            "generated_count": 0,
            "fallback_count": len(pending_explanations),
            "resumed_count": int(resumed_count),
            "requested_count": int(len(pending_rows)),
            "total_count": int(len(generation_rows)),
            "generation_mode": "deterministic_fallback",
            "fallback_reason": "missing_explain_model_path",
            "explain_model_path": "",
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
            "explain_model_load_mode": "not_requested",
            "explanation_sidecar_path": str(sidecar_path or ""),
            "explain_quality_gate": bool(quality_gate),
            "resume_rejected_count": int(resume_rejected_count),
            "resume_rejected_reasons": dict(resume_rejected_reasons),
            "quality_rejected_count": 0,
            "quality_rejected_reasons": {},
        }
    tokenizer = None
    model = None
    explain_model_load_mode = "inline_reload"
    if generation_runtime is not None:
        if generation_runtime.get("load_error"):
            pending_explanations = _deterministic_explanations(pending_rows)
            for position, row, prompt_hash, explanation in zip(pending_positions, pending_rows, pending_hashes, pending_explanations):
                explanations[position] = explanation
                if sidecar_path:
                    _append_jsonl(
                        sidecar_path,
                        [
                            {
                                "node_id": int(row["node_id"]),
                                "component_name": component_name,
                                "prompt_role": row.get("prompt_role", ""),
                                "prompt_hash": prompt_hash,
                                "explanation": explanation,
                                "generation_mode": "deterministic_fallback",
                            }
                        ],
                    )
            return explanations, {
                "generated_count": 0,
                "fallback_count": len(pending_explanations),
                "resumed_count": int(resumed_count),
                "requested_count": int(len(pending_rows)),
                "total_count": int(len(generation_rows)),
                "generation_mode": "deterministic_fallback",
                "fallback_reason": "local_explain_model_unavailable",
                "explain_model_error": str(generation_runtime.get("load_error")),
                "explain_model_path": str(args.explain_model_path),
                "explain_batch_size": int(args.explain_batch_size),
                "explain_max_input_length": int(args.explain_max_input_length),
                "explain_max_new_tokens": int(args.explain_max_new_tokens),
                "explain_model_load_mode": "load_failed_once",
                "explanation_sidecar_path": str(sidecar_path or ""),
                "explain_quality_gate": bool(quality_gate),
                "resume_rejected_count": int(resume_rejected_count),
                "resume_rejected_reasons": dict(resume_rejected_reasons),
                "quality_rejected_count": 0,
                "quality_rejected_reasons": {},
            }
        tokenizer = generation_runtime.get("tokenizer")
        model = generation_runtime.get("model")
        explain_model_load_mode = "single_load_per_run"
    if tokenizer is None or model is None:
        try:
            tokenizer, model = _load_generation_model(args, device)
        except Exception as exc:
            if bool(getattr(args, "explain_required", False)):
                raise RuntimeError(
                    "Failed to load the required local explain model. "
                    "Refusing to write a deterministic-fallback explanation cache."
                ) from exc
            pending_explanations = _deterministic_explanations(pending_rows)
            for position, row, prompt_hash, explanation in zip(pending_positions, pending_rows, pending_hashes, pending_explanations):
                explanations[position] = explanation
                if sidecar_path:
                    _append_jsonl(
                        sidecar_path,
                        [
                            {
                                "node_id": int(row["node_id"]),
                                "component_name": component_name,
                                "prompt_role": row.get("prompt_role", ""),
                                "prompt_hash": prompt_hash,
                                "explanation": explanation,
                                "generation_mode": "deterministic_fallback",
                            }
                        ],
                    )
            return explanations, {
                "generated_count": 0,
                "fallback_count": len(pending_explanations),
                "resumed_count": int(resumed_count),
                "requested_count": int(len(pending_rows)),
                "total_count": int(len(generation_rows)),
                "generation_mode": "deterministic_fallback",
                "fallback_reason": "local_explain_model_unavailable",
                "explain_model_error": str(exc),
                "explain_model_path": str(args.explain_model_path),
                "explain_batch_size": int(args.explain_batch_size),
                "explain_max_input_length": int(args.explain_max_input_length),
                "explain_max_new_tokens": int(args.explain_max_new_tokens),
                "explain_model_load_mode": "load_failed",
                "explanation_sidecar_path": str(sidecar_path or ""),
                "explain_quality_gate": bool(quality_gate),
                "resume_rejected_count": int(resume_rejected_count),
                "resume_rejected_reasons": dict(resume_rejected_reasons),
                "quality_rejected_count": 0,
                "quality_rejected_reasons": {},
            }
    if any(item is None for item in prompts):
        prompts = [_format_chat_prompt(tokenizer, row["system"], row["user"]) for row in generation_rows]
        prompt_hashes = [_string_sha256(prompt) for prompt in prompts]
        if sidecar_path:
            resumed_by_key, resume_rejected_reasons = _load_resume_explanations(
                sidecar_path,
                component_name,
                quality_gate=quality_gate,
            )
        else:
            resumed_by_key, resume_rejected_reasons = {}, {}
        resume_rejected_count = int(sum(int(value) for value in resume_rejected_reasons.values()))
        explanations = [None for _ in generation_rows]
        resumed_count = 0
        pending_rows = []
        pending_hashes = []
        pending_positions = []
        for idx, (row, prompt_hash) in enumerate(zip(generation_rows, prompt_hashes)):
            key = (int(row["node_id"]), str(prompt_hash))
            if key in resumed_by_key:
                explanations[idx] = resumed_by_key[key]
                resumed_count += 1
            else:
                pending_rows.append(row)
                pending_hashes.append(prompt_hash)
                pending_positions.append(idx)
    pending_prompts = [prompts[position] for position in pending_positions]
    fallback_count = 0
    generated_count = 0
    quality_rejected = defaultdict(int)
    start_time = time.time()
    log_every = max(int(getattr(args, "explain_log_every", 50) or 0), 0)
    effective_batch_size = max(int(_effective_explain_batch_size(args, device)), 1)
    last_logged_generated = 0
    if not pending_rows and resumed_count:
        return explanations, {
            "generated_count": 0,
            "fallback_count": 0,
            "resumed_count": int(resumed_count),
            "requested_count": 0,
            "total_count": int(len(generation_rows)),
            "generation_mode": "llm_generation_resumed",
            "explain_model_path": str(args.explain_model_path),
            "explain_batch_size": int(args.explain_batch_size),
            "effective_explain_batch_size": int(effective_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
            "explain_model_load_mode": explain_model_load_mode,
            "explanation_sidecar_path": str(sidecar_path or ""),
            "explain_quality_gate": bool(quality_gate),
            "resume_rejected_count": int(resume_rejected_count),
            "resume_rejected_reasons": dict(resume_rejected_reasons),
            "quality_rejected_count": 0,
            "quality_rejected_reasons": {},
        }
    try:
        with torch.no_grad():
            for start in range(0, len(pending_prompts), effective_batch_size):
                prompt_batch = pending_prompts[start : start + effective_batch_size]
                batch_rows = pending_rows[start : start + effective_batch_size]
                batch_positions = pending_positions[start : start + effective_batch_size]
                batch_hashes = pending_hashes[start : start + effective_batch_size]
                batch = tokenizer(
                    prompt_batch,
                    padding=True,
                    truncation=True,
                    max_length=int(args.explain_max_input_length),
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model.generate(
                    **batch,
                    max_new_tokens=int(args.explain_max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                batch_sidecar_rows = []
                for row_idx, generated_ids in enumerate(outputs):
                    continuation = decoder_generation_continuation(generated_ids, batch["input_ids"][row_idx])
                    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                    if not text:
                        fallback_count += 1
                        text = _fallback_explanation_text(batch_rows[row_idx])
                    issue = _explanation_quality_issue(text) if quality_gate else ""
                    if issue:
                        quality_rejected[issue] += 1
                        if bool(getattr(args, "explain_required", False)):
                            preview = " ".join(str(text or "").split())[:240]
                            raise RuntimeError(
                                f"Generated explanation failed quality gate for component={component_name}, "
                                f"node_id={int(batch_rows[row_idx]['node_id'])}, reason={issue}, preview={preview!r}"
                            )
                        fallback_count += 1
                        text = _fallback_explanation_text(batch_rows[row_idx])
                    explanations[batch_positions[row_idx]] = text
                    generated_count += 1
                    batch_sidecar_rows.append(
                        {
                            "node_id": int(batch_rows[row_idx]["node_id"]),
                            "component_name": component_name,
                            "prompt_role": batch_rows[row_idx].get("prompt_role", ""),
                            "prompt_hash": batch_hashes[row_idx],
                            "explanation": text,
                            "generation_mode": "llm_generation",
                        }
                    )
                if sidecar_path:
                    _append_jsonl(sidecar_path, batch_sidecar_rows)
                completed_count = int(resumed_count + generated_count)
                should_log = bool(log_every) and (
                    completed_count == int(len(generation_rows))
                    or generated_count - last_logged_generated >= log_every
                )
                if should_log:
                    last_logged_generated = int(generated_count)
                    elapsed = max(time.time() - start_time, 1e-6)
                    rows_per_second = float(generated_count) / elapsed
                    progress_payload = {
                        "component": component_name,
                        "completed": completed_count,
                        "generated": int(generated_count),
                        "resumed": int(resumed_count),
                        "total": int(len(generation_rows)),
                        "pending_total": int(len(pending_rows)),
                        "rows_per_second": rows_per_second,
                        "elapsed_seconds": float(elapsed),
                    }
                    print({"status": "explain_progress", **progress_payload})
                    _wandb_log(
                        wandb_run,
                        {
                            f"precompute/explain/{component_name}/completed": completed_count,
                            f"precompute/explain/{component_name}/generated_so_far": int(generated_count),
                            f"precompute/explain/{component_name}/resumed": int(resumed_count),
                            f"precompute/explain/{component_name}/total": int(len(generation_rows)),
                            f"precompute/explain/{component_name}/rows_per_second": rows_per_second,
                            f"precompute/explain/{component_name}/elapsed_seconds": float(elapsed),
                        },
                        step=int(wandb_step_base + completed_count),
                    )
    except Exception as exc:
        if bool(getattr(args, "explain_required", False)):
            raise RuntimeError(
                "Required LLM explanation generation failed. "
                "Refusing to write a deterministic-fallback explanation cache."
            ) from exc
        pending_explanations = _deterministic_explanations(pending_rows)
        for position, row, prompt_hash, explanation in zip(pending_positions, pending_rows, pending_hashes, pending_explanations):
            if explanations[position] is not None:
                continue
            explanations[position] = explanation
            if sidecar_path:
                _append_jsonl(
                    sidecar_path,
                    [
                        {
                            "node_id": int(row["node_id"]),
                            "component_name": component_name,
                            "prompt_role": row.get("prompt_role", ""),
                            "prompt_hash": prompt_hash,
                            "explanation": explanation,
                            "generation_mode": "deterministic_fallback",
                        }
                    ],
                )
        return explanations, {
            "generated_count": int(generated_count),
            "fallback_count": int(len(pending_explanations)),
            "resumed_count": int(resumed_count),
            "requested_count": int(len(pending_rows)),
            "total_count": int(len(generation_rows)),
            "generation_mode": "deterministic_fallback",
            "fallback_reason": "generation_failed",
            "explain_model_error": str(exc),
            "explain_model_path": str(args.explain_model_path),
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
            "explain_model_load_mode": explain_model_load_mode,
                "explanation_sidecar_path": str(sidecar_path or ""),
                "explain_quality_gate": bool(quality_gate),
                "resume_rejected_count": int(resume_rejected_count),
                "resume_rejected_reasons": dict(resume_rejected_reasons),
                "quality_rejected_count": int(sum(int(value) for value in quality_rejected.values())),
                "quality_rejected_reasons": dict(quality_rejected),
            }
    finally:
        if generation_runtime is None and "model" in locals():
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {
        "generated_count": int(generated_count),
        "fallback_count": int(fallback_count),
        "resumed_count": int(resumed_count),
        "requested_count": int(len(pending_rows)),
        "total_count": int(len(generation_rows)),
        "elapsed_seconds": float(time.time() - start_time),
        "generation_mode": "llm_generation",
        "explain_model_path": str(args.explain_model_path),
        "explain_batch_size": int(args.explain_batch_size),
        "effective_explain_batch_size": int(effective_batch_size),
        "explain_max_input_length": int(args.explain_max_input_length),
        "explain_max_new_tokens": int(args.explain_max_new_tokens),
        "explain_model_load_mode": explain_model_load_mode,
        "explanation_sidecar_path": str(sidecar_path or ""),
        "explain_quality_gate": bool(quality_gate),
        "resume_rejected_count": int(resume_rejected_count),
        "resume_rejected_reasons": dict(resume_rejected_reasons),
        "quality_rejected_count": int(sum(int(value) for value in quality_rejected.values())),
        "quality_rejected_reasons": dict(quality_rejected),
    }
    return explanations, summary


def _attach_component_explanations(prompt_bundle, component_name, explanations):
    generation_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get(component_name, []))
    if generation_rows and len(explanations) != len(generation_rows):
        raise ValueError(
            f"Generated {component_name} explanation count does not match the number of generation prompts."
        )
    row_by_node = {int(row["node_id"]): row for row in prompt_bundle.get("prompt_rows", [])}
    if component_name in prompt_bundle.get("selected_components", []):
        prompt_bundle["prompt_components"][component_name] = [
            _expert_embedding_prompt(component_name, item) for item in explanations
        ]
    for generation_row, explanation in zip(generation_rows, explanations):
        row = row_by_node.get(int(generation_row["node_id"]))
        if row is None:
            continue
        row[f"{component_name}_explanation"] = explanation
        row[f"{component_name}_embedding_prompt"] = _expert_embedding_prompt(component_name, explanation)
    return prompt_bundle


def _prepare_v2_conflict_generation_rows(prompt_bundle):
    generation_rows = []
    for row in prompt_bundle.get("prompt_rows", []):
        if "conflict" not in row.get("selected_components", []):
            continue
        generation_prompt = _conflict_explain_generation_prompt(row)
        row["conflict_generation_prompt"] = generation_prompt["user"]
        generation_rows.append(
            {
                "node_id": int(row["node_id"]),
                "component_name": "conflict",
                "system": generation_prompt["system"],
                "user": generation_prompt["user"],
                "fallback_explanation": generation_prompt["fallback_explanation"],
                "prompt_role": row.get("component_prompt_roles", {}).get("conflict", "cross_view_conflict_explainer"),
            }
        )
    prompt_bundle.setdefault("generation_rows_by_component", {})["conflict"] = generation_rows
    return prompt_bundle


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute GLANCE-style, relation-aware, or expert prompt caches.")
    parser.add_argument("--dataset", type=str, default="TwiBot-20")
    parser.add_argument("--graph_data_variant", choices=GRAPH_DATA_VARIANT_CHOICES, default="labeled")
    parser.add_argument("--context_graph_variant", choices=GRAPH_DATA_VARIANT_CHOICES, default=None)
    parser.add_argument("--center_node_scope", choices=CENTER_NODE_SCOPE_CHOICES, default="labeled")
    parser.add_argument(
        "--routed_nodes_path",
        type=Path,
        default=None,
        help=(
            "Optional explicit routed-node index file (.jsonl/.json/.pt/.txt). "
            "When set, prompt-expert construction, explain generation, and encoding run only on those graph-global node ids."
        ),
    )
    parser.add_argument("--text_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="prompt_precompute")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_QWEN_MODEL_PATH,
        help=(
            "Local embedding-model path, cached HF repo id, or alias. Legacy prompt modes default to Qwen3; "
            "expert_* + --prompt_family_version v2 defaults to the SimTeG finetuned RoBERTa alias "
            "'roberta_finetuned' (yzxjb/roberta-finetuned-20) when this flag is omitted."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--finetuned_roberta_checkpoint_path",
        type=Path,
        default=None,
        help=(
            "Optional SimTeG LM checkpoint (.pkl) used when --model_path resolves to roberta_finetuned. "
            "If omitted, v2 expert modes search for TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_mode", choices=PROMPT_MODE_CHOICES, default="glance_concat_ego_hop1_hop2")
    parser.add_argument("--prompt_family_version", choices=PROMPT_FAMILY_VERSION_CHOICES, default="v1")
    parser.add_argument("--neighbor_sampling_policy", choices=NEIGHBOR_SAMPLING_CHOICES, default="auto")
    parser.add_argument("--selection_embedding_path", type=Path, default=None)
    parser.add_argument("--support_selection_embedding_path", type=Path, default=None)
    parser.add_argument("--tweet_source_mode", choices=TWEET_SOURCE_MODE_CHOICES, default="norm_user_text")
    parser.add_argument("--node_source_path", type=Path, default=None)
    parser.add_argument("--edge_source_path", type=Path, default=None)
    parser.add_argument("--tweet_sample_size", type=int, default=6)
    parser.add_argument("--tweet_clean_level", choices=TWEET_CLEAN_LEVEL_CHOICES, default="light")
    parser.add_argument("--tweet_keep_hashtag_surface", type=_parse_bool, default=True)
    parser.add_argument("--tweet_keep_emoji_surface", type=_parse_bool, default=True)
    parser.add_argument("--neighbor_cap", type=int, default=5)
    parser.add_argument("--following_quota", type=int, default=3)
    parser.add_argument("--follower_quota", type=int, default=3)
    parser.add_argument("--max_length_ego", type=int, default=1024)
    parser.add_argument("--max_length_hop", type=int, default=4096)
    parser.add_argument("--normalize", type=_parse_bool, default=True)
    parser.add_argument("--save_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--explain_model_path", type=str, default=None)
    parser.add_argument("--explain_trust_remote_code", action="store_true")
    parser.add_argument(
        "--explain_required",
        action="store_true",
        help=(
            "Fail instead of falling back to deterministic explanations when the local explain model "
            "is missing or generation fails. Use this for real LLM-as-explainer experiments."
        ),
    )
    parser.add_argument(
        "--explain_batch_size",
        type=int,
        default=0,
        help="Batch size for LLM explanation generation. Use 0 for auto (CUDA=2, CPU=1).",
    )
    parser.add_argument("--explain_max_input_length", type=int, default=2048)
    parser.add_argument("--explain_max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--explain_log_every",
        type=int,
        default=50,
        help="Log and wandb-report explanation generation progress every N newly generated rows per component.",
    )
    parser.add_argument(
        "--explain_quality_gate",
        type=_parse_bool,
        default=True,
        help=(
            "Validate explanation sidecars and fresh LLM outputs before reuse. "
            "When enabled, empty, punctuation-only, or prompt-echo explanations are rejected and regenerated."
        ),
    )
    parser.add_argument(
        "--explain_component_cache_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-component explanation JSONL sidecars. "
            "Existing rows with matching node_id and prompt hash are reused for resumable precompute."
        ),
    )
    return parser


def run(args):
    _apply_prompt_expert_v2_encoder_defaults(args)
    wandb_run = _setup_precompute_wandb(args)
    if getattr(args, "routed_nodes_path", None) and not str(getattr(args, "prompt_mode", "")).startswith("expert_"):
        raise ValueError(
            "--routed_nodes_path is only supported for expert_* prompt modes because the saved cache "
            "keeps full-graph row shape with zero-filled unselected nodes for refiner-only consumption."
        )
    dataset_path = resolve_dataset_path(args.dataset)
    _wandb_log(
        wandb_run,
        {
            "precompute/status": 0,
            "precompute/stage_code": 0,
            "precompute/prompt_mode": str(args.prompt_mode),
            "precompute/prompt_family_version": str(getattr(args, "prompt_family_version", "v1")),
        },
        step=0,
    )
    prompt_family_version = str(getattr(args, "prompt_family_version", "v1")).lower()
    graph_variant_value = getattr(args, "context_graph_variant", None) or getattr(args, "graph_data_variant", "labeled")
    variant_paths = _resolve_graph_variant_paths(dataset_path, graph_variant_value, args.text_path)
    texts, resolved_text_path = _load_texts(dataset_path, variant_paths["text_path"])
    if not texts:
        raise ValueError("No node texts found for Qwen prompt precompute.")

    labels_path = dataset_path / "labels.pt"
    labels = torch.load(labels_path, map_location="cpu")
    labeled_node_count = int(labels.shape[0]) if torch.is_tensor(labels) and labels.dim() >= 1 else int(len(labels))
    graph_data_variant = variant_paths["variant"]
    full_node_count = int(len(texts))
    target_node_bundle = _resolve_target_node_ids(args, labeled_node_count=labeled_node_count, full_node_count=full_node_count)
    target_node_ids = list(target_node_bundle["target_node_ids"])
    target_node_scope = str(target_node_bundle["target_node_scope"])
    center_node_scope_requested = str(target_node_bundle["center_node_scope_requested"])
    target_node_source = str(target_node_bundle["target_node_source"])
    raw_tweet_index_scope = str(target_node_bundle["raw_tweet_index_scope"])
    routed_nodes_path = str(target_node_bundle["routed_nodes_path"])
    if torch.is_tensor(labels) and labels.dim() > 1:
        class_count = int(labels.shape[1])
        classes = ["human", "bot"] if class_count == 2 else [f"class_{idx}" for idx in range(class_count)]
    else:
        classes = sorted(int(x) for x in torch.unique(torch.as_tensor(labels)).tolist())

    edge_index = torch.load(variant_paths["edge_index_path"], map_location="cpu")
    edge_type = torch.load(variant_paths["edge_type_path"], map_location="cpu")
    selection_feature_bundle = None
    if str(args.neighbor_sampling_policy).lower() == "center_induced_relation_aware":
        selection_feature_bundle = _resolve_selection_feature_bundle(
            args,
            dataset_path,
            graph_variant=graph_data_variant,
            graph_node_count=full_node_count,
            labeled_node_count=labeled_node_count,
        )
    prompt_bundle = _resolve_prompt_bundle(
        args,
        texts,
        edge_index,
        edge_type,
        classes,
        target_node_ids=target_node_ids,
        selection_features=selection_feature_bundle["features"] if selection_feature_bundle is not None else None,
    )
    _wandb_log(
        wandb_run,
        {
            "precompute/stage_code": 1,
            "precompute/target_node_count": int(len(target_node_ids)),
            "precompute/full_graph_node_count": int(full_node_count),
            "precompute/selected_component_count": int(len(prompt_bundle.get("selected_components", []))),
            "precompute/generation_component_count": int(len(prompt_bundle.get("generation_rows_by_component", {}))),
        },
        step=1,
    )

    tweet_source_mode_requested = str(getattr(args, "tweet_source_mode", "norm_user_text")).lower()
    raw_tweet_bundle = None
    raw_tweet_stats = {
        "raw_tweet_user_coverage": 0.0,
        "raw_tweet_fallback_count": 0,
        "raw_tweet_covered_count": 0,
    }
    tweet_source_mode_effective = "norm_user_text"
    if str(args.prompt_mode).startswith("expert_") and tweet_source_mode_requested == "raw_post_edges" and _selected_components_require_tweet(prompt_bundle):
        raw_tweet_bundle = _build_raw_tweet_source_bundle(
            args,
            dataset_path,
            graph_data_variant=graph_data_variant,
            center_node_scope=raw_tweet_index_scope,
            target_node_ids=target_node_ids,
        )
        tweet_source_mode_effective = "raw_post_edges"

    embedding_model_source = _require_local_pretrained_source(args.model_path, model_role="Embedding model")
    embedding_encoder_tag = _infer_embedding_encoder_tag(embedding_model_source)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else _default_output_path(
            dataset_path,
            str(args.prompt_mode),
            prompt_family_version=prompt_family_version,
            embedding_encoder_tag=embedding_encoder_tag,
        )
    )
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or a different --output_path.")
    ensure_dir(output_path.parent)
    explanation_cache_dir = (
        Path(args.explain_component_cache_dir)
        if getattr(args, "explain_component_cache_dir", None)
        else output_path.parent
    )
    ensure_dir(explanation_cache_dir)

    explain_summary_by_component = {}
    explain_rows_for_sidecar = []
    component_explain_sidecar_paths = {}
    device = _device_from_args(str(args.device))
    if str(args.prompt_mode).startswith("expert_"):
        if raw_tweet_bundle is not None:
            records = [_parse_norm_user_record(raw_text, idx) for idx, raw_text in enumerate(texts)]
            target_records = [records[idx] for idx in target_node_ids]
            target_records, raw_tweet_stats = _attach_raw_tweet_records(target_records, raw_tweet_bundle, args)
            for item in target_records:
                records[int(item["node_id"])] = item
            prompt_bundle = _resolve_expert_prompt_bundle(
                args,
                records,
                edge_index,
                edge_type,
                target_node_ids=target_node_ids,
                selection_features=selection_feature_bundle["features"] if selection_feature_bundle is not None else None,
            )
        else:
            tweet_source_mode_effective = "norm_user_text"
        generation_runtime = _resolve_generation_runtime(args, device)
        initial_generation_order = ["ego", "graph_following", "graph_follower", "tweet"]
        for component_name in initial_generation_order:
            generation_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get(component_name, []))
            if not generation_rows:
                continue
            component_sidecar_path = (
                explanation_cache_dir / f"{output_path.stem}_{component_name}_explanations.jsonl"
            )
            component_explain_sidecar_paths[component_name] = str(component_sidecar_path)
            explanations, component_summary = _generate_component_explanations(
                args,
                generation_rows,
                device,
                generation_runtime=generation_runtime,
                component_name=component_name,
                sidecar_path=component_sidecar_path,
                wandb_run=wandb_run,
                wandb_step_base=1000 * (1 + len(explain_summary_by_component)),
            )
            explain_summary_by_component[component_name] = component_summary
            prompt_bundle = _attach_component_explanations(prompt_bundle, component_name, explanations)
            _wandb_log(
                wandb_run,
                {
                    "precompute/explain_component_done": 1,
                    f"precompute/explain/{component_name}/rows": int(len(generation_rows)),
                    f"precompute/explain/{component_name}/generated_count": int(component_summary.get("generated_count", 0)),
                    f"precompute/explain/{component_name}/fallback_count": int(component_summary.get("fallback_count", 0)),
                    f"precompute/explain/{component_name}/generation_mode_code": 1
                    if str(component_summary.get("generation_mode", "")) == "llm_generation"
                    else 0,
                },
                step=10 + len(explain_summary_by_component),
            )
            explain_rows_for_sidecar.extend(
                {
                    "node_id": int(row["node_id"]),
                    "component_name": component_name,
                    "prompt_role": row.get("prompt_role", ""),
                    "explanation": explanation,
                }
                for row, explanation in zip(generation_rows, explanations)
            )
        if bool(prompt_bundle.get("needs_conflict_generation")):
            prompt_bundle = _prepare_v2_conflict_generation_rows(prompt_bundle)
        conflict_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get("conflict", []))
        if conflict_rows:
            component_sidecar_path = (
                explanation_cache_dir / f"{output_path.stem}_conflict_explanations.jsonl"
            )
            component_explain_sidecar_paths["conflict"] = str(component_sidecar_path)
            explanations, component_summary = _generate_component_explanations(
                args,
                conflict_rows,
                device,
                generation_runtime=generation_runtime,
                component_name="conflict",
                sidecar_path=component_sidecar_path,
                wandb_run=wandb_run,
                wandb_step_base=5000,
            )
            explain_summary_by_component["conflict"] = component_summary
            prompt_bundle = _attach_component_explanations(prompt_bundle, "conflict", explanations)
            _wandb_log(
                wandb_run,
                {
                    "precompute/explain_component_done": 1,
                    "precompute/explain/conflict/rows": int(len(conflict_rows)),
                    "precompute/explain/conflict/generated_count": int(component_summary.get("generated_count", 0)),
                    "precompute/explain/conflict/fallback_count": int(component_summary.get("fallback_count", 0)),
                    "precompute/explain/conflict/generation_mode_code": 1
                    if str(component_summary.get("generation_mode", "")) == "llm_generation"
                    else 0,
                },
                step=20,
            )
            explain_rows_for_sidecar.extend(
                {
                    "node_id": int(row["node_id"]),
                    "component_name": "conflict",
                    "prompt_role": row.get("prompt_role", ""),
                    "explanation": explanation,
                }
                for row, explanation in zip(conflict_rows, explanations)
            )
        if generation_runtime is not None:
            generation_runtime["model"] = None
            generation_runtime["tokenizer"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tokenizer, model, resolved_embedding_model_source, embedding_pooling_mode, simteg_checkpoint_summary = _load_embedding_model(args, device)
    embedding_encoder_tag = _infer_embedding_encoder_tag(resolved_embedding_model_source)
    encoded = {}
    component_effective_budget = {}
    for component_name, component_prompts in prompt_bundle["prompt_components"].items():
        if not component_prompts:
            continue
        length_group = prompt_bundle["component_max_length_group"].get(component_name, "hop")
        max_length = int(args.max_length_ego) if length_group == "ego" else int(args.max_length_hop)
        component_effective_budget[component_name] = _component_effective_token_budget(
            tokenizer,
            model,
            max_length,
            pooling_mode=embedding_pooling_mode,
        )
        encoded[component_name] = _encode_texts(
            model,
            tokenizer,
            component_prompts,
            device,
            int(args.batch_size),
            max_length,
            bool(args.normalize),
            embedding_pooling_mode,
        )
        _wandb_log(
            wandb_run,
            {
                "precompute/encode_component_done": 1,
                f"precompute/encode/{component_name}/rows": int(encoded[component_name].shape[0]),
                f"precompute/encode/{component_name}/dim": int(encoded[component_name].shape[1])
                if encoded[component_name].dim() == 2
                else 0,
            },
            step=100 + len(encoded),
        )
    for component_name, values in prompt_bundle.get("structured_components", {}).items():
        if not values:
            continue
        encoded[component_name] = torch.tensor(values, dtype=torch.float32)
        _wandb_log(
            wandb_run,
            {
                "precompute/structured_component_done": 1,
                f"precompute/structured/{component_name}/rows": int(encoded[component_name].shape[0]),
                f"precompute/structured/{component_name}/dim": int(encoded[component_name].shape[1])
                if encoded[component_name].dim() == 2
                else 0,
            },
            step=150 + len(encoded),
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    selected_components = [item for item in prompt_bundle["selected_components"] if item in encoded]
    if not selected_components:
        raise ValueError(f"Prompt mode {args.prompt_mode} did not produce any encoded components.")
    if len(selected_components) == 1:
        selected_embeddings = encoded[selected_components[0]]
        embedding_mode = "single_component"
    else:
        selected_embeddings = torch.cat([encoded[name] for name in selected_components], dim=1)
        embedding_mode = "selected_component_concat"

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    target_index_tensor = torch.tensor(target_node_ids, dtype=torch.long)
    target_mask_tensor = torch.zeros((int(full_node_count),), dtype=torch.bool)
    if int(target_index_tensor.numel()) > 0:
        target_mask_tensor[target_index_tensor] = True
    full_graph_selected_embeddings = _scatter_selected_tensor_to_full_graph(
        selected_embeddings.to(save_dtype).contiguous(),
        target_index_tensor,
        full_node_count,
    )
    payload = {
        "embeddings": full_graph_selected_embeddings,
        "prompt_family": prompt_bundle["prompt_family"],
        "semantic_view_mode": prompt_bundle.get("semantic_view_mode", str(args.prompt_mode)),
        "active_components": list(selected_components),
        "component_order": list(selected_components),
        "graph_data_variant": graph_data_variant,
        "context_graph_variant": graph_data_variant,
        "target_node_count": int(len(target_node_ids)),
        "full_graph_node_count": int(full_node_count),
        "labeled_node_count": int(labeled_node_count),
        "target_node_ids": target_index_tensor.clone(),
        "target_node_mask": target_mask_tensor.clone(),
        "target_node_scope": str(target_node_scope),
        "target_node_source": str(target_node_source),
        "center_node_scope_requested": str(center_node_scope_requested),
        "routed_nodes_path": str(routed_nodes_path),
        "prompt_family_version": prompt_family_version,
        "component_prompt_roles": dict(prompt_bundle.get("component_prompt_roles", {})),
        "structured_component_schema": dict(prompt_bundle.get("structured_component_schema", {})),
        "embedding_encoder_tag": embedding_encoder_tag,
        "tweet_source_mode_requested": tweet_source_mode_requested,
        "tweet_source_mode_effective": tweet_source_mode_effective,
    }
    for component_name, component_tensor in encoded.items():
        payload[component_name] = _scatter_selected_tensor_to_full_graph(
            component_tensor.to(save_dtype).contiguous(),
            target_index_tensor,
            full_node_count,
        )
    if prompt_bundle["prompt_family"] in {"prompt_expert_bundle_v1", "prompt_expert_bundle_center_induced_v1"}:
        for scalar_key, scalar_values in prompt_bundle["scalar_features"].items():
            payload[scalar_key] = _scatter_selected_tensor_to_full_graph(
                torch.tensor(scalar_values, dtype=torch.float32),
                target_index_tensor,
                full_node_count,
            )
    elif prompt_bundle["prompt_family"] == "prompt_expert_bundle_v2":
        for scalar_key, scalar_values in prompt_bundle["scalar_features"].items():
            payload[scalar_key] = _scatter_selected_tensor_to_full_graph(
                torch.tensor(scalar_values, dtype=torch.float32),
                target_index_tensor,
                full_node_count,
            )
    write_torch(output_path, payload)
    component_cache_paths = {}
    if prompt_bundle["prompt_family"] in {"prompt_expert_bundle_v1", "prompt_expert_bundle_v2", "prompt_expert_bundle_center_induced_v1"}:
        for component_name in selected_components:
            if component_name not in payload:
                continue
            component_payload = {
                key: value
                for key, value in payload.items()
                if key not in set(EXPERT_COMPONENT_NAMES_WITH_STRUCTURED)
            }
            component_payload["embeddings"] = payload[component_name]
            component_payload[component_name] = payload[component_name]
            component_payload["active_components"] = [component_name]
            component_payload["component_order"] = [component_name]
            component_payload["selected_embedding_keys"] = [component_name]
            component_payload["selected_embedding_dim"] = int(payload[component_name].shape[1]) if payload[component_name].dim() == 2 else 0
            component_payload["embedding_mode"] = "single_component_cache"
            component_payload["source_concat_cache_path"] = str(output_path)
            component_path = output_path.with_name(f"{output_path.stem}_{component_name}.pt")
            write_torch(component_path, component_payload)
            component_cache_paths[component_name] = str(component_path)

    tensor_hashes = {"embeddings": tensor_sha256(payload["embeddings"])}
    tensor_hashes["target_node_ids"] = tensor_sha256(payload["target_node_ids"])
    tensor_hashes["target_node_mask"] = tensor_sha256(payload["target_node_mask"])
    for component_name in encoded:
        tensor_hashes[component_name] = tensor_sha256(payload[component_name])
    for scalar_key in prompt_bundle.get("scalar_features", {}):
        tensor_hashes[scalar_key] = tensor_sha256(payload[scalar_key])

    component_budget = {}
    for component_name, group in prompt_bundle["component_max_length_group"].items():
        component_budget[component_name] = int(args.max_length_ego) if group == "ego" else int(args.max_length_hop)
    if not component_effective_budget:
        component_effective_budget = {
            name: _component_effective_token_budget(tokenizer, model, budget, pooling_mode=embedding_pooling_mode)
            for name, budget in component_budget.items()
        }

    prompt_sidecar_path = output_path.with_name(f"{output_path.stem}_prompts.jsonl")
    _write_jsonl(prompt_sidecar_path, prompt_bundle.get("prompt_rows", []))
    explain_sidecar_path = None
    if explain_rows_for_sidecar:
        if prompt_family_version == "v2":
            explain_sidecar_path = output_path.with_name(f"{output_path.stem}_explanations.jsonl")
        else:
            explain_sidecar_path = output_path.with_name("glance_prompt_expert_ego_explain.jsonl")
        _write_jsonl(explain_sidecar_path, explain_rows_for_sidecar)

    generation_mode = "not_used"
    explain_model_path_manifest = ""
    explain_model_load_mode_manifest = "not_used"
    generation_modes = sorted(
        {
            str(summary.get("generation_mode", "unknown"))
            for summary in explain_summary_by_component.values()
        }
    )
    explain_model_load_modes = sorted(
        {
            str(summary.get("explain_model_load_mode", "unknown"))
            for summary in explain_summary_by_component.values()
        }
    )
    if explain_summary_by_component:
        generation_mode = generation_modes[0] if len(generation_modes) == 1 else "mixed_component_generation_modes"
        explain_model_path_manifest = str(next(iter(explain_summary_by_component.values())).get("explain_model_path", ""))
        explain_model_load_mode_manifest = (
            explain_model_load_modes[0] if len(explain_model_load_modes) == 1 else "mixed_component_load_modes"
        )

    manifest = {
        "status": "completed",
        "dataset": str(args.dataset),
        "dataset_path": str(dataset_path),
        "graph_data_variant": graph_data_variant,
        "context_graph_variant": graph_data_variant,
        "center_node_scope": str(target_node_scope),
        "center_node_scope_requested": str(center_node_scope_requested),
        "target_node_source": str(target_node_source),
        "routed_nodes_path": str(routed_nodes_path),
        "text_path": str(resolved_text_path),
        "edge_index_path": str(variant_paths["edge_index_path"]),
        "edge_type_path": str(variant_paths["edge_type_path"]),
        "output_path": str(output_path),
        "component_cache_paths": dict(component_cache_paths),
        "prompt_family_version": prompt_family_version,
        "embedding_model_path": str(resolved_embedding_model_source),
        "embedding_encoder_tag": embedding_encoder_tag,
        "embedding_pooling_mode": embedding_pooling_mode,
        "embedding_encoder_contract": (
            "frozen_simteg_finetuned_roberta_lm_checkpoint"
            if simteg_checkpoint_summary
            else "simteg_finetuned_roberta_text_encoder"
            if embedding_pooling_mode == "simteg_mean"
            else "generic_huggingface_embedding_encoder"
        ),
        "finetuned_roberta_checkpoint_path": str(
            (simteg_checkpoint_summary or {}).get("checkpoint_path", getattr(args, "finetuned_roberta_checkpoint_path", "") or "")
        ),
        "finetuned_roberta_checkpoint_load": dict(simteg_checkpoint_summary or {}),
        "embedding_tokenization_contract": (
            "SimTeG-compatible: padding=True, truncation=True, max_length<=512, add_special_tokens=False, final hidden-state plain mean"
            if embedding_pooling_mode == "simteg_mean"
            else "padding=True, truncation=True, encoder-aligned pooling"
        ),
        "generation_mode": generation_mode,
        "explain_model_load_mode": explain_model_load_mode_manifest,
        "explain_model_path": explain_model_path_manifest,
        "explain_required": bool(getattr(args, "explain_required", False)),
        "explain_batch_size": int(getattr(args, "explain_batch_size", 1)),
        "effective_explain_batch_size": int(_effective_explain_batch_size(args, device)),
        "explain_max_input_length": int(getattr(args, "explain_max_input_length", 2048)),
        "explain_max_new_tokens": int(getattr(args, "explain_max_new_tokens", 128)),
        "explain_log_every": int(getattr(args, "explain_log_every", 50)),
        "explain_component_cache_dir": str(explanation_cache_dir),
        "component_explanation_sidecar_paths": dict(component_explain_sidecar_paths),
        "model_path": str(args.model_path),
        "resolved_model_path": str(resolved_embedding_model_source),
        "trust_remote_code": bool(args.trust_remote_code),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "prompt_mode": str(args.prompt_mode),
        "prompt_family": prompt_bundle["prompt_family"],
        "prompt_style": prompt_bundle["prompt_style"],
        "tweet_source_mode_requested": tweet_source_mode_requested,
        "tweet_source_mode_effective": tweet_source_mode_effective,
        "node_source_path": str(raw_tweet_bundle["node_source_path"]) if raw_tweet_bundle is not None else "",
        "edge_source_path": str(raw_tweet_bundle["edge_source_path"]) if raw_tweet_bundle is not None else "",
        "post_edge_direction": str(raw_tweet_bundle["post_edge_direction"]) if raw_tweet_bundle is not None else "",
        "tweet_clean_level": str(getattr(args, "tweet_clean_level", "light")),
        "tweet_sample_size": int(getattr(args, "tweet_sample_size", 6)),
        "tweet_keep_hashtag_surface": bool(getattr(args, "tweet_keep_hashtag_surface", True)),
        "tweet_keep_emoji_surface": bool(getattr(args, "tweet_keep_emoji_surface", True)),
        "raw_tweet_user_coverage": float(raw_tweet_stats.get("raw_tweet_user_coverage", 0.0)),
        "raw_tweet_fallback_count": int(raw_tweet_stats.get("raw_tweet_fallback_count", 0)),
        "raw_tweet_covered_count": int(raw_tweet_stats.get("raw_tweet_covered_count", 0)),
        "component_order": list(selected_components),
        "component_max_length_group": dict(prompt_bundle["component_max_length_group"]),
        "neighbor_sampling_policy": prompt_bundle["neighbor_sample_policy"],
        "neighbor_sampling_policy_requested": prompt_bundle["neighbor_sampling_policy_requested"],
        "selection_embedding_path": str(selection_feature_bundle["labeled_path"]) if selection_feature_bundle is not None else "",
        "support_selection_embedding_path": str(selection_feature_bundle["support_path"]) if selection_feature_bundle is not None else "",
        "selection_embedding_mode": str(selection_feature_bundle["mode"]) if selection_feature_bundle is not None else "",
        "neighbor_cap": int(args.neighbor_cap),
        "following_quota": int(args.following_quota),
        "follower_quota": int(args.follower_quota),
        "max_length_ego": int(args.max_length_ego),
        "max_length_hop": int(args.max_length_hop),
        "component_prompt_token_budget": component_budget,
        "component_effective_token_budget": component_effective_budget,
        "normalize": bool(args.normalize),
        "save_dtype": str(args.save_dtype),
        "limit": int(args.limit),
        "target_node_count": int(len(target_node_ids)),
        "full_graph_node_count": int(full_node_count),
        "labeled_node_count": int(labeled_node_count),
        "target_node_preview": [int(item) for item in target_node_ids[:32]],
        "target_node_membership_encoding": "payload_target_node_ids_and_bool_mask",
        "payload_row_layout": "full_graph_zero_fill_for_unselected_nodes" if len(target_node_ids) < int(full_node_count) else "full_graph_dense",
        "semantic_mode": str(args.prompt_mode),
        "semantic_view_mode": prompt_bundle.get("semantic_view_mode", str(args.prompt_mode)),
        "component_prompt_roles": dict(prompt_bundle.get("component_prompt_roles", {})),
        "embedding_keys": list(encoded.keys()),
        "selected_embedding_keys": selected_components,
        "selected_embedding_dim": int(payload["embeddings"].shape[1]),
        "component_embedding_dim": int(next(iter(encoded.values())).shape[1]),
        "embedding_dtype": str(payload["embeddings"].dtype),
        "num_nodes": int(payload["embeddings"].shape[0]),
        "embedding_mode": embedding_mode,
        "prompt_cap_characters_per_node": 2000,
        "prompt_classes": classes,
        "tensor_sha256": tensor_hashes,
        "text_sha256": _hash_strings(texts),
        "neighbor_counts": {key: int(np.sum(values)) for key, values in prompt_bundle["counts"].items()},
        "directional_quota": prompt_bundle.get("directional_quota"),
        "support_contrast_quota": prompt_bundle.get("support_contrast_quota"),
        "prompt_sidecar_path": str(prompt_sidecar_path),
        "explanation_sidecar_path": str(explain_sidecar_path) if explain_sidecar_path else "",
        "ego_explain_sidecar_path": str(explain_sidecar_path) if explain_sidecar_path else "",
        "edge_type_semantics": {
            "relation_1": "following source->target",
            "relation_0": "follower source->target interpreted as target receives source as follower",
                "source": "aligned to LLMbot.trainer_glance directional semantic-view helper",
        },
        "strict_stage_compatibility": {
            "payload_embeddings_key": "embeddings",
            "payload_component_keys": list(encoded.keys()),
            "payload_scalar_keys": list(prompt_bundle.get("scalar_features", {}).keys()),
            "semantic_view_mode": prompt_bundle.get("semantic_view_mode", str(args.prompt_mode)),
            "target_node_scope": str(target_node_scope),
            "target_node_membership_encoding": "payload_target_node_ids_and_bool_mask",
            "payload_row_layout": "full_graph_zero_fill_for_unselected_nodes" if len(target_node_ids) < int(full_node_count) else "full_graph_dense",
            "support_fill_policy": "labeled_prefix_only" if target_node_scope == "labeled" and graph_data_variant == "full_graph_support" else "none",
            "same_root_requirement": (
                "joint_router_refinement may consume this prompt cache through "
                "--joint_refiner_embedding_path while keeping backbone provenance pinned "
                "to the current run's graph_detector_prepare artifact."
            ),
        },
        "note": (
            "glance_* modes preserve GLANCE-style prompt families; relation_aware_* modes use "
            "direction-aware social-context prompts with deterministic heuristic neighbor selection; "
            "expert_* modes emit structured prompt-expert bundles for strict refiner-only semantic overrides. "
            "v1 preserves the older mixed embedding/explanation behavior, while v2 upgrades expert prompts to an explanation-first pipeline with encoder-aware naming and pooling. "
            "Explanation generation stays offline-first: it reuses a local explain-model snapshot when available and otherwise falls back to deterministic summaries instead of blocking on HuggingFace downloads."
        ),
    }
    if explain_summary_by_component:
        manifest["expert_explain_generation_by_component"] = explain_summary_by_component
        if "ego" in explain_summary_by_component:
            manifest["ego_explain_generation"] = explain_summary_by_component["ego"]
    write_json(output_path.with_name(f"{output_path.stem}_manifest.json"), manifest)
    _wandb_log(
        wandb_run,
        {
            "precompute/stage_code": 2,
            "precompute/completed": 1,
            "precompute/num_nodes": int(payload["embeddings"].shape[0]),
            "precompute/embedding_dim": int(payload["embeddings"].shape[1]),
            "precompute/target_node_count": int(len(target_node_ids)),
            "precompute/selected_embedding_dim": int(payload["embeddings"].shape[1]),
            "precompute/generation_mode_code": 1 if generation_mode == "llm_generation" else 0,
        },
        step=999,
    )
    _finish_precompute_wandb(wandb_run, exit_code=0)
    print(
        {
            "status": "completed",
            "output_path": str(output_path),
            "prompt_mode": str(args.prompt_mode),
            "num_nodes": int(payload["embeddings"].shape[0]),
            "embedding_dim": int(payload["embeddings"].shape[1]),
            "dtype": str(payload["embeddings"].dtype),
        }
    )
    return manifest


def main():
    args = build_parser().parse_args()
    try:
        run(args)
    except BaseException:
        _finish_precompute_wandb(exit_code=1)
        raise


if __name__ == "__main__":
    main()
