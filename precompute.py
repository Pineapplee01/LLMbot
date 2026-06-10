"""Precompute prompt caches for GLANCE-style, relation-aware, and expert views.

This helper builds per-node prompt texts, optionally generates explanation-first
expert summaries, encodes them with a local embedding model, applies
encoder-aligned pooling plus optional l2 normalization, and stores a prompt
cache under a stable ``embeddings`` key for downstream graph_detector_prepare
and strict GLANCE runs. HyperScan-style KNN and hypergraph helpers live in
``hypergnn.py`` so this file stays prompt-cache focused.
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

from hypergnn import (
    resolve_selection_feature_bundle,
    select_center_induced_directional_neighbors,
)
from prompt import (
    _conflict_explain_generation_prompt as build_conflict_explain_generation_prompt,
    _conflict_prompt as build_conflict_prompt,
    _conflict_prompt_partitioned as build_conflict_prompt_partitioned,
    _ego_botsay_tweet_metadata_prompt as build_ego_botsay_tweet_metadata_prompt,
    _ego_explain_generation_prompt as build_ego_explain_generation_prompt,
    _expert_embedding_prompt as build_expert_embedding_prompt,
    _evidence_card_generation_system as build_evidence_card_generation_system,
    _evidence_card_prompt_rules as build_evidence_card_prompt_rules,
    _graph_explain_generation_prompt as build_graph_explain_generation_prompt,
    _graph_prompt as build_graph_prompt,
    _graph_prompt_partitioned as build_graph_prompt_partitioned,
    _mhlgc_llm_guide_prompt as build_mhlgc_llm_guide_prompt,
    _mhlgc_semantic_embedding_prompt as build_mhlgc_semantic_embedding_prompt,
    _summary_generation_system as build_summary_generation_system,
    _summary_prompt_rules as build_summary_prompt_rules,
    _tweet_explain_generation_prompt as build_tweet_explain_generation_prompt,
    _tweet_prompt as build_tweet_prompt,
)
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
ULTRATAG_PROMPT_MODES = (
    "ultratag_s_subgraph_v1",
)
RESIDUAL_AUDIT_PROMPT_MODES = (
    "residual_audit_v1",
)
DGP_PROMPT_MODES = (
    "dgp_predictor_v1",
    "dgp_predictor_v2",
)
MHLGC_PROMPT_MODES = (
    "mhlgc_llm_guide",
)
PROMPT_MODE_CHOICES = (
    LEGACY_PROMPT_MODES
    + EXPERT_PROMPT_MODES
    + ULTRATAG_PROMPT_MODES
    + RESIDUAL_AUDIT_PROMPT_MODES
    + DGP_PROMPT_MODES
    + MHLGC_PROMPT_MODES
)
PROMPT_FAMILY_VERSION_CHOICES = ("v1", "v2", "v3")
EXPLAIN_PROMPT_STYLE_CHOICES = ("default", "botsay")
RESIDUAL_PROMPT_VARIANT_CHOICES = ("base_as_hypothesis", "no_base", "base_as_assertion")
DGP_PROMPT_VARIANT_CHOICES = (
    "target_fine_neighbor_coarse",
    "target_only",
    "norm_text_following_summary",
    "norm_text_follower_summary",
    "norm_text_following_follower_summary",
    "norm_text_target_only",
)
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


def _is_botsay_prompt_role(prompt_role):
    return "botsay" in str(prompt_role or "").strip().lower()


def _postprocess_botsay_explanation(text):
    text = str(text or "").strip()
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    match = re.search(
        r"(?is)label\s*:\s*(bot|human)\b(.*?)explanation\s*:\s*(.+)",
        normalized,
    )
    if not match:
        return text
    label = match.group(1).strip().lower()
    explanation = match.group(3).strip()
    lower_explanation = explanation.lower()
    role_line_match = re.search(r"(?im)(?:^|\n)\s*(system|user|assistant)\s*:?\s*(?:\n|$)", explanation)
    role_line_cut = role_line_match.start() if role_line_match else -1
    stop_markers = (
        "\nlabel:",
        "\nexplanation:",
        "label:",
        "explanation:",
        "the following task focuses on evaluating whether a twitter user is a bot or human",
        "\ntarget user:",
        "target user:",
        "\ntarget user profile:",
        "target user profile:",
        "\ntweet behavior summary:",
        "tweet behavior summary:",
        "\ntweet evidence:",
        "tweet evidence:",
        "\nrelation summary:",
        "relation summary:",
        "\nthese users follow the target user:",
        "these users follow the target user:",
        "\nthe target user follows these users:",
        "the target user follows these users:",
        "\nfollowing-side evidence:",
        "following-side evidence:",
        "\nfollower-side evidence:",
        "follower-side evidence:",
        "\ncross-view hints:",
        "cross-view hints:",
        "\noutput format:",
        "output format:",
        "\nuse only the provided evidence",
        "use only the provided evidence",
        "\nyou should output the label first and explanation after.",
        "you should output the label first and explanation after.",
    )
    cut_positions = [lower_explanation.find(marker) for marker in stop_markers if lower_explanation.find(marker) >= 0]
    if role_line_cut >= 0:
        cut_positions.append(int(role_line_cut))
    if cut_positions:
        explanation = explanation[: min(cut_positions)].strip()
    explanation = re.sub(r"\n{3,}", "\n\n", explanation).strip(" \n\t:-")
    if not explanation:
        return text
    return f"Label: {label}\nExplanation: {explanation}"


def _postprocess_generated_explanation(text, prompt_role=""):
    if _is_botsay_prompt_role(prompt_role):
        return _postprocess_botsay_explanation(text)
    return _sanitize_generated_explanation(text)


def _sanitize_generated_explanation(text):
    text = str(text or "").strip()
    if not text:
        return ""
    # Qwen occasionally emits an isolated chat role token on its own line before
    # continuing the explanation. Remove that role marker without discarding
    # the surrounding natural-language evidence.
    text = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(" \n\t:-")


def _explanation_quality_issue(text):
    text = _sanitize_generated_explanation(text)
    if not text:
        return "empty"
    lowered = text.lower()
    special_token_markers = ("</s>", "<s>", "<|im_", "<|endoftext|>", "<|end")
    if any(marker in lowered for marker in special_token_markers):
        return "special_token_echo"
    prompt_tag_markers = (
        "<target_text",
        "</target_text",
        "<neighbor_text",
        "</neighbor_text",
        "<question>",
        "</question>",
        "<answer>",
        "</answer>",
    )
    if any(marker in lowered for marker in prompt_tag_markers):
        return "prompt_echo"
    placeholder_count = len(re.findall(r"@USER\b", text))
    word_count = max(len(re.findall(r"\S+", text)), 1)
    if placeholder_count >= 20 and placeholder_count / word_count >= 0.25:
        return "placeholder_repetition"
    compact = "".join(text.split())
    if compact and compact.count("!") > max(30, int(len(compact) * 0.2)):
        return "many_exclamation_marks"
    hard_echo_markers = (
        "write 4-6 evidence-grounded sentences",
        "do not output only a label",
        "system\nyou analyze twitter accounts",
    )
    if any(marker in lowered for marker in hard_echo_markers):
        return "prompt_echo"
    if re.search(r"(?im)^\s*(system|user|assistant)\s*:\s+", text):
        return "prompt_echo"
    if len(text) < 24 or len(re.findall(r"[A-Za-z]", text)) < 12:
        return "too_short"
    return ""


def _is_explanation_quality_ok(text):
    return not _explanation_quality_issue(text)


def _quality_retry_user_prompt(user_prompt, issue, attempt):
    return (
        f"{str(user_prompt or '').strip()}\n\n"
        "Regenerate the answer because the previous continuation failed the text-quality gate "
        f"({issue}). Return only 2-4 plain English evidence-grounded sentences. "
        "Output in English even when the supplied account text is non-English, multilingual, or noisy. "
        "Do not copy raw non-English spans, XML tags, chat role names, special tokens, prompt instructions, "
        "or repeated @USER placeholders. If the evidence is sparse or not reliably interpretable, state that "
        "plainly in English and summarize only observable metadata, activity, and text-quality cues."
    ).strip()


def _generate_single_explanation_text(model, tokenizer, prompt, args, device):
    batch = tokenizer(
        [prompt],
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
    continuation = decoder_generation_continuation(outputs[0], batch["input_ids"][0])
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


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


def _select_routed_node_split(raw, split_name="all"):
    split = str(split_name or "all").strip().lower()
    if isinstance(raw, dict):
        for wrapper_key in ("routed_nodes", "splits", "split_nodes", "selected_nodes"):
            nested = raw.get(wrapper_key)
            if isinstance(nested, dict):
                try:
                    return _select_routed_node_split(nested, split_name=split)
                except ValueError:
                    pass
    if split in {"", "all", "union", "target", "targets"}:
        if isinstance(raw, dict):
            for key in ("total", "all", "union", "target_node_ids", "routed_node_ids", "routed_nodes", "nodes"):
                if key in raw and not isinstance(raw.get(key), dict):
                    return raw[key]
            combined = []
            for key in ("train", "valid", "validation", "val", "test"):
                if key in raw:
                    combined.extend(_coerce_node_id_sequence(raw[key]))
            if combined:
                return combined
        return raw
    if not isinstance(raw, dict):
        raise ValueError("--routed_nodes_split requires a mapping-style routed node file.")
    aliases = {
        "train": ("train", "train_nodes", "train_node_ids", "routed_train", "train_routed_nodes"),
        "valid": ("valid", "validation", "val", "valid_nodes", "validation_nodes", "valid_node_ids", "routed_valid"),
        "val": ("valid", "validation", "val", "valid_nodes", "validation_nodes", "valid_node_ids", "routed_valid"),
        "test": ("test", "test_nodes", "test_node_ids", "routed_test", "test_routed_nodes"),
    }
    keys = aliases.get(split)
    if keys is None:
        raise ValueError(f"Unsupported --routed_nodes_split {split_name!r}; expected all/train/valid/test.")
    for key in keys:
        if key in raw:
            return raw[key]
    if "node_ids" in raw and isinstance(raw.get("split_counts"), dict):
        ordered_split_names = ("train", "valid", "test")
        counts = {str(key).lower(): int(value) for key, value in raw["split_counts"].items()}
        if split == "val":
            split = "valid"
        if split in counts:
            start = 0
            for name in ordered_split_names:
                count = int(counts.get(name, 0))
                end = start + count
                if name == split:
                    node_ids = _coerce_node_id_sequence(raw["node_ids"])
                    return node_ids[start:end]
                start = end
    raise ValueError(
        f"Could not find split {split_name!r} in routed node mapping. Available keys: {sorted(raw.keys())}"
    )


def _load_explicit_target_node_ids_for_split(path: Path, split_name="all"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = read_json(path, default=None)
        values = _coerce_node_id_sequence(_select_routed_node_split(raw, split_name=split_name))
    elif suffix == ".pt":
        raw = torch.load(path, map_location="cpu")
        values = _coerce_node_id_sequence(_select_routed_node_split(raw, split_name=split_name))
    else:
        if str(split_name or "all").strip().lower() not in {"", "all", "union", "target", "targets"}:
            raise ValueError("--routed_nodes_split is only supported for .json or .pt routed-node files.")
        values = _load_explicit_target_node_ids(path)
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
        explanation = _sanitize_generated_explanation(row.get("explanation", ""))
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
    routed_nodes_split = str(getattr(args, "routed_nodes_split", "all") or "all").strip().lower()
    target_node_source = "center_node_scope"
    raw_tweet_index_scope = center_node_scope_requested
    if routed_nodes_path:
        target_node_ids = _load_explicit_target_node_ids_for_split(Path(routed_nodes_path), split_name=routed_nodes_split)
        if not target_node_ids:
            raise ValueError(
                f"--routed_nodes_path did not yield any node ids for split={routed_nodes_split}: {routed_nodes_path}"
            )
        invalid = [int(item) for item in target_node_ids if int(item) < 0 or int(item) >= int(full_node_count)]
        if invalid:
            preview = invalid[:10]
            raise ValueError(
                f"--routed_nodes_path contains node ids outside [0, {int(full_node_count) - 1}]: "
                f"{preview}{' ...' if len(invalid) > len(preview) else ''}"
            )
        target_node_scope = "explicit_routed_nodes"
        target_node_source = f"routed_nodes_path:{routed_nodes_split}"
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
        "routed_nodes_split": str(routed_nodes_split),
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


def _target_account_text_for_llm(record, tweet_stats, *, include_identity=True, brief=False):
    lines = [
        "PROFILE:",
        _profile_cue_summary(record, include_identity=include_identity),
        "TWEET_BEHAVIOR:",
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
    ]
    rendered = "\n".join(lines)
    limit = 1400 if brief else 2200
    return _truncate_chars(rendered, limit) or "No usable account text was available."


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


def _coverage_bucket(count):
    count = int(count or 0)
    if count <= 0:
        return "sparse"
    if count <= 2:
        return "partial"
    return "rich"


def _structured_card_lines(
    *,
    observed_bot_like_cues=None,
    observed_human_like_cues=None,
    relation_ambiguity=None,
    possible_benign_explanation=None,
    evidence_coverage="partial",
    evidence_consistency="mixed",
    source_support="partly",
):
    def _items(values):
        values = [str(item).strip().lstrip("- ").strip() for item in (values or []) if str(item).strip()]
        return values or ["None"]

    lines = [
        "OBSERVED_BOT_LIKE_CUES:",
        *[f"- {item}" for item in _items(observed_bot_like_cues)],
        "OBSERVED_HUMAN_LIKE_CUES:",
        *[f"- {item}" for item in _items(observed_human_like_cues)],
        "RELATION_AMBIGUITY:",
        *[f"- {item}" for item in _items(relation_ambiguity)],
        "POSSIBLE_BENIGN_EXPLANATION:",
        *[f"- {item}" for item in _items(possible_benign_explanation)],
        f"EVIDENCE_COVERAGE: {str(evidence_coverage)}",
        f"EVIDENCE_CONSISTENCY: {str(evidence_consistency)}",
        f"SUPPORTED_BY_SOURCE: {str(source_support)}",
    ]
    return "\n".join(lines)


def _deterministic_ego_evidence_card_fallback(record):
    bot_like = []
    human_like = []
    ambiguity = []
    benign = []
    if not record.get("bio_present"):
        bot_like.append("Profile bio is missing or empty.")
    if record.get("verified", False):
        human_like.append("Account is marked verified in the profile metadata.")
    if record.get("protected", False):
        human_like.append("Account is protected, limiting observable public evidence.")
    if record.get("display_name") and record.get("screen_name"):
        human_like.append("Display name and screen name are present.")
    if record.get("missing_profile_score", None) is None:
        ambiguity.append("Profile completeness must be inferred from sparse serialized fields.")
    benign.append("Missing profile fields can reflect privacy choices or incomplete data collection.")
    return _structured_card_lines(
        observed_bot_like_cues=bot_like,
        observed_human_like_cues=human_like,
        relation_ambiguity=ambiguity,
        possible_benign_explanation=benign,
        evidence_coverage="partial",
        evidence_consistency="mixed",
        source_support="partly",
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


def _deterministic_tweet_evidence_card_fallback(record, tweet_stats):
    bot_like = []
    human_like = []
    ambiguity = []
    benign = []
    tweet_count = int(tweet_stats.get("tweet_count", 0) or 0)
    if float(tweet_stats.get("rt_ratio", 0.0)) >= 0.6:
        bot_like.append(f"Retweet ratio is high ({float(tweet_stats['rt_ratio']):.2f}).")
    if max(float(tweet_stats.get("url_ratio", 0.0)), float(tweet_stats.get("hashtag_ratio", 0.0))) >= 0.5:
        bot_like.append("Posting sample contains heavy URL or hashtag usage.")
    if float(tweet_stats.get("duplicate_ratio", 0.0)) >= 0.3:
        bot_like.append(f"Duplicate ratio is elevated ({float(tweet_stats['duplicate_ratio']):.2f}).")
    if tweet_count > 0 and float(tweet_stats.get("duplicate_ratio", 0.0)) < 0.3:
        human_like.append("Sampled tweets are not dominated by duplicates.")
    if tweet_count == 0:
        ambiguity.append("No sampled tweets are available for posting-behavior evidence.")
    else:
        ambiguity.append("Tweet sample is limited and may not represent long-term behavior.")
    benign.append("Retweets, links, or hashtags can reflect normal news sharing, fandom, or campaign participation.")
    return _structured_card_lines(
        observed_bot_like_cues=bot_like,
        observed_human_like_cues=human_like,
        relation_ambiguity=ambiguity,
        possible_benign_explanation=benign,
        evidence_coverage=_coverage_bucket(tweet_count),
        evidence_consistency="mixed" if bot_like and human_like else ("consistent" if bot_like or human_like else "mixed"),
        source_support="partly" if tweet_count else "weak",
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


def _deterministic_graph_evidence_card_fallback(direction_name, total_count, reciprocal_ratio_selected, neighbor_records):
    bot_like = []
    human_like = []
    ambiguity = []
    benign = []
    direction_label = "following" if direction_name == "following" else "follower"
    total_count = int(total_count or 0)
    if total_count <= 0:
        ambiguity.append(f"No {direction_label} neighbors are available in this directed view.")
    else:
        if float(reciprocal_ratio_selected) <= 0.1:
            ambiguity.append(f"Selected {direction_label} neighbors have low reciprocity ({float(reciprocal_ratio_selected):.2f}).")
        else:
            human_like.append(f"Selected {direction_label} view has reciprocal evidence ({float(reciprocal_ratio_selected):.2f}).")
        top_neighbor = neighbor_records[0] if neighbor_records else None
        if top_neighbor is not None:
            human_like.append(
                f"Visible neighbor context includes {top_neighbor['display_name'] or 'unknown'} (@{top_neighbor['screen_name'] or 'unknown'})."
            )
    if direction_name == "following":
        ambiguity.append("High similarity among followed accounts can indicate either camouflage or an organic interest community.")
        benign.append("Accounts can follow semantically similar users because of normal topical interests.")
    else:
        ambiguity.append("Suspicious or similar followers can indicate amplification, purchased audience, or victimization of the center account.")
        benign.append("The center account may receive automated followers without controlling them.")
    return _structured_card_lines(
        observed_bot_like_cues=bot_like,
        observed_human_like_cues=human_like,
        relation_ambiguity=ambiguity,
        possible_benign_explanation=benign,
        evidence_coverage=_coverage_bucket(total_count),
        evidence_consistency="mixed",
        source_support="partly" if total_count else "weak",
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


def _deterministic_conflict_evidence_card_fallback(row):
    hints = [item.lstrip("- ").strip() for item in (row.get("conflict_mismatch_hints", []) or []) if str(item).strip()]
    bot_like = []
    human_like = []
    ambiguity = []
    benign = []
    for hint in hints[:3]:
        if "retweet" in hint.lower() or "promotion" in hint.lower() or "sparse profile" in hint.lower():
            bot_like.append(hint)
        else:
            ambiguity.append(hint)
    if not hints:
        ambiguity.append("Profile, tweet, following, and follower views expose no strong explicit mismatch cue.")
    benign.append("Cross-view mismatch can be caused by sparse data, benign topical focus, or incomplete graph coverage.")
    return _structured_card_lines(
        observed_bot_like_cues=bot_like,
        observed_human_like_cues=human_like,
        relation_ambiguity=ambiguity,
        possible_benign_explanation=benign,
        evidence_coverage="partial",
        evidence_consistency="conflicting" if bot_like or ambiguity else "mixed",
        source_support="partly",
    )


def _summary_generation_system():
    return build_summary_generation_system()


def _evidence_card_generation_system():
    return build_evidence_card_generation_system()


def _summary_prompt_rules():
    return build_summary_prompt_rules()


def _evidence_card_prompt_rules():
    return build_evidence_card_prompt_rules()


def _expert_embedding_prompt(component_name, explanation_text, evidence_schema="summary"):
    return build_expert_embedding_prompt(component_name, explanation_text, evidence_schema=evidence_schema)


def _tweet_explain_generation_prompt(record, tweet_stats, evidence_schema="summary", prompt_style="default"):
    structured = str(evidence_schema) == "structured_evidence_card"
    prompt = build_tweet_explain_generation_prompt(
        _profile_cue_summary(record, include_identity=True),
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
        evidence_schema=evidence_schema,
        prompt_style=prompt_style,
    )
    return {
        "system": prompt["system"],
        "user": prompt["user"],
        "fallback_explanation": (
            _deterministic_tweet_evidence_card_fallback(record, tweet_stats)
            if structured
            else _deterministic_tweet_explain_fallback(record, tweet_stats)
        ),
        "prompt_role": prompt["prompt_role"],
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


def _graph_explain_generation_prompt(direction_name, ego_record, neighbor_records, summary, evidence_schema="summary", prompt_style="default"):
    structured = str(evidence_schema) == "structured_evidence_card"
    prompt = build_graph_explain_generation_prompt(
        direction_name,
        _profile_cue_summary(ego_record, include_identity=True),
        [_neighbor_card_v2(item) for item in neighbor_records],
        summary,
        evidence_schema=evidence_schema,
        prompt_style=prompt_style,
    )
    return {
        "system": prompt["system"],
        "user": prompt["user"],
        "fallback_explanation": (
            _deterministic_graph_evidence_card_fallback(
                direction_name,
                summary["count"],
                summary["reciprocal_ratio_selected"],
                neighbor_records,
            )
            if structured
            else _deterministic_graph_explain_fallback(
                direction_name,
                summary["count"],
                summary["reciprocal_ratio_selected"],
                neighbor_records,
            )
        ),
        "prompt_role": prompt["prompt_role"],
    }


def _conflict_explain_generation_prompt(row, evidence_schema="summary", prompt_style="default"):
    structured = str(evidence_schema) == "structured_evidence_card"
    mismatch_hints = list(row.get("conflict_mismatch_hints", []) or [])
    prompt = build_conflict_explain_generation_prompt(
        row.get("profile_cue_summary", ""),
        row.get("tweet_explanation", "None"),
        row.get("graph_following_explanation", "None"),
        row.get("graph_follower_explanation", "None"),
        mismatch_hints,
        evidence_schema=evidence_schema,
        prompt_style=prompt_style,
    )
    return {
        "system": prompt["system"],
        "user": prompt["user"],
        "fallback_explanation": (
            _deterministic_conflict_evidence_card_fallback(row)
            if structured
            else _deterministic_conflict_explain_fallback(row)
        ),
        "prompt_role": prompt["prompt_role"],
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
        if "qwen2.5" in token or "qwen25" in token or "qwen2-5" in token:
            return "qwen25"
        if "qwen3" in token:
            return "qwen3"
        return "qwen"
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


def _is_prompt_expert_explanation_first(args):
    return str(getattr(args, "prompt_family_version", "v1")).lower() in {"v2", "v3"} and str(
        getattr(args, "prompt_mode", "")
    ).startswith("expert_")


def _has_cli_option(raw_argv, option_name):
    prefix = f"{option_name}="
    return any(item == option_name or str(item).startswith(prefix) for item in raw_argv)


def _apply_prompt_expert_explanation_first_encoder_defaults(args, raw_argv=None):
    raw_argv = list(sys.argv[1:] if raw_argv is None else raw_argv)
    is_ultratag = str(getattr(args, "prompt_mode", "")) in ULTRATAG_PROMPT_MODES
    if not _is_prompt_expert_explanation_first(args) and not is_ultratag:
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
    prompt_role = str(row.get("prompt_role", "") or "")
    if "dgp_v2" in prompt_role:
        direction = _dgp_v2_relation_label(row.get("summary_direction", "following"))
        neighbor_id = row.get("neighbor_node_id", None)
        neighbor_part = (
            f" for selected neighbor {int(neighbor_id)}"
            if neighbor_id is not None and str(neighbor_id) != ""
            else ""
        )
        return (
            f"The {direction} evidence{neighbor_part} was too noisy or multilingual "
            "for a reliable generated English summary. Treat this as weak coarse "
            "neighbor context and rely primarily on the target account evidence."
        )
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


def _mhlgc_relation_tags(node_id, candidate_id, context):
    node_id = int(node_id)
    candidate_id = int(candidate_id)
    tags = []
    if candidate_id in context["following"][node_id]:
        tags.append("target_follows_candidate")
    if candidate_id in context["follower"][node_id]:
        tags.append("candidate_follows_target")
    if (
        candidate_id in context["following"][node_id]
        and candidate_id in context["follower"][node_id]
    ):
        tags.append("mutual")
    if not tags and candidate_id in context["undirected"][node_id]:
        tags.append("undirected_relation_1hop")
    if not tags:
        tags.append("semantic_knn_only")
    return tags


def _mhlgc_neighbor_card(node_id, candidate_id, texts, context, similarity=None, rank=None):
    candidate_id = int(candidate_id)
    card = {
        "node_id": candidate_id,
        "relation_to_target": _mhlgc_relation_tags(node_id, candidate_id, context),
        "text": _node_text(texts[candidate_id]),
    }
    if rank is not None:
        card["rank"] = int(rank)
    if similarity is not None:
        card["semantic_similarity"] = float(similarity)
    return card


def _mhlgc_original_view(node_id, texts, context, following_quota, follower_quota):
    following_ranked = _rank_directional_candidates(
        node_id,
        context["following"][node_id],
        texts,
        context,
    )
    follower_ranked = _rank_directional_candidates(
        node_id,
        context["follower"][node_id],
        texts,
        context,
    )
    following_ids = following_ranked[: int(following_quota)]
    follower_ids = follower_ranked[: int(follower_quota)]
    mutual_count = len(set(context["following"][node_id]).intersection(set(context["follower"][node_id])))
    return {
        "view_type": "original_directed_relation_view",
        "node_id": int(node_id),
        "counts": {
            "following": int(len(set(context["following"][node_id]))),
            "follower": int(len(set(context["follower"][node_id]))),
            "mutual": int(mutual_count),
            "undirected_1hop": int(len(context["undirected"][node_id])),
        },
        "selected_following_neighbors": [
            _mhlgc_neighbor_card(node_id, candidate, texts, context, rank=rank)
            for rank, candidate in enumerate(following_ids, start=1)
        ],
        "selected_follower_neighbors": [
            _mhlgc_neighbor_card(node_id, candidate, texts, context, rank=rank)
            for rank, candidate in enumerate(follower_ids, start=1)
        ],
        "selection_policy": "directional_heuristic_fixed_quota",
    }


def _mhlgc_hypergraph_view(node_id, texts, context, selection_features, knn_k):
    if selection_features is None:
        raise ValueError("mhlgc_llm_guide requires selection_features from --selection_embedding_path.")
    features = selection_features.detach().cpu().float()
    if features.dim() != 2:
        raise ValueError("mhlgc_llm_guide selection features must be a 2-D tensor.")
    if not (0 <= int(node_id) < int(features.shape[0])):
        raise ValueError(f"mhlgc_llm_guide node_id {int(node_id)} is outside selection feature rows.")
    k = max(int(knn_k), 1)
    center = F.normalize(features[int(node_id)].view(1, -1), p=2, dim=1, eps=1e-12)
    all_features = F.normalize(features, p=2, dim=1, eps=1e-12)
    scores = torch.matmul(all_features, center.t()).view(-1)
    topk = min(int(k) + 1, int(scores.numel()))
    values, indices = torch.topk(scores, k=topk, largest=True)
    members = []
    for rank, (candidate, similarity) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        candidate = int(candidate)
        if candidate == int(node_id):
            continue
        members.append(
            _mhlgc_neighbor_card(
                node_id,
                candidate,
                texts,
                context,
                similarity=float(similarity),
                rank=len(members) + 1,
            )
        )
        if len(members) >= k:
            break
    return {
        "view_type": "hyperscan_style_knn_hypergraph_view",
        "node_id": int(node_id),
        "hyperedge_center": int(node_id),
        "construction": "one center-induced semantic KNN hyperedge over selection embeddings",
        "knn_k": int(k),
        "members": members,
        "member_count": int(len(members)),
    }


def _resolve_mhlgc_prompt_bundle(args, texts, edge_index, edge_type, target_node_ids=None, selection_features=None):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    prompts = {"mhlgc_llm_guide": []}
    prompt_rows = []
    counts = {
        "original_following_nodes": [],
        "original_follower_nodes": [],
        "hypergraph_member_nodes": [],
    }
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    for node_id in target_ids:
        node_id = int(node_id)
        original_view = _mhlgc_original_view(
            node_id,
            texts,
            context,
            following_quota=int(args.following_quota),
            follower_quota=int(args.follower_quota),
        )
        hypergraph_view = _mhlgc_hypergraph_view(
            node_id,
            texts,
            context,
            selection_features,
            knn_k=int(args.neighbor_cap),
        )
        target_text = _node_text(texts[node_id])
        guide_prompt = build_mhlgc_llm_guide_prompt(
            node_id=node_id,
            original_view=original_view,
            hypergraph_view=hypergraph_view,
            target_text=target_text,
            role="borderline_anchor",
        )
        embedding_prompt = build_mhlgc_semantic_embedding_prompt(
            node_id=node_id,
            original_view=original_view,
            hypergraph_view=hypergraph_view,
            target_text=target_text,
            role="borderline_anchor",
        )
        prompts["mhlgc_llm_guide"].append(embedding_prompt)
        prompt_rows.append(
            {
                "node_id": int(node_id),
                "component_name": "mhlgc_llm_guide",
                "prompt_role": guide_prompt["prompt_role"],
                "prompt_family": guide_prompt["prompt_family"],
                "system": guide_prompt["system"],
                "user": guide_prompt["user"],
                "prompt": embedding_prompt,
                "target_text": target_text,
                "original_view": original_view,
                "hypergraph_view": hypergraph_view,
            }
        )
        counts["original_following_nodes"].append(
            int(len(original_view["selected_following_neighbors"]))
        )
        counts["original_follower_nodes"].append(
            int(len(original_view["selected_follower_neighbors"]))
        )
        counts["hypergraph_member_nodes"].append(int(hypergraph_view["member_count"]))
    return {
        "prompt_family": "mhlgc_llm_guide_v1",
        "prompt_style": "mhlgc_original_relation_plus_hyperscan_knn_hypergraph",
        "prompt_family_version": "mhlgc_llm_guide_v1",
        "evidence_schema": "llm_guided_multiview_semantic_embedding",
        "evidence_card_fields": [],
        "prompt_components": prompts,
        "structured_components": {},
        "structured_component_schema": {},
        "generation_rows_by_component": {},
        "component_max_length_group": {"mhlgc_llm_guide": "hop"},
        "component_prompt_roles": {"mhlgc_llm_guide": "mhlgc_multiview_llm_guide"},
        "counts": counts,
        "neighbor_sample_policy": "mhlgc_original_relation_plus_selection_embedding_knn_hypergraph",
        "directional_quota": {"following": int(args.following_quota), "follower": int(args.follower_quota)},
        "support_contrast_quota": {},
        "component_order": {"mhlgc_llm_guide": ["mhlgc_llm_guide"]},
        "selected_components": ["mhlgc_llm_guide"],
        "scalar_features": {},
        "prompt_rows": prompt_rows,
        "semantic_view_mode": "mhlgc_llm_guide_v1",
        "selection_policy_note": (
            "MH-LGC-style prompt cache serializes the original directed relation view and one "
            "HyperScan-style semantic KNN hypergraph view. The prompt asks for LLM guide embeddings "
            "for hard-negative contrastive learning and forbids final bot/human labels."
        ),
    }


def _residual_class_names(classes):
    if isinstance(classes, (list, tuple)) and len(classes) == 2:
        if all(isinstance(item, str) for item in classes):
            return [str(classes[0]), str(classes[1])]
        if set(int(item) for item in classes) == {0, 1}:
            return ["human", "bot"]
    return [str(item) for item in classes]


def _load_residual_base_outputs(path: Path, target_node_ids):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--residual_base_outputs_path does not exist: {path}")
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"Expected base outputs dict at {path}, got {type(raw).__name__}.")

    def _first_tensor(*keys):
        for key in keys:
            value = raw.get(key)
            if torch.is_tensor(value):
                return value.detach().cpu()
        return None

    logits = _first_tensor("logits", "all_logits", "output_logits")
    prob = _first_tensor("prob", "probs", "probabilities", "softmax", "all_prob")
    pred = _first_tensor("pred", "preds", "prediction", "predictions", "all_pred")
    if prob is None and logits is not None:
        prob = torch.softmax(logits.float(), dim=-1)
    if pred is None and prob is not None:
        pred = torch.argmax(prob.float(), dim=-1)
    if pred is not None and pred.dim() > 1:
        pred = torch.argmax(pred.float(), dim=-1)
    if prob is None and logits is None and pred is None:
        raise ValueError(
            f"Base outputs at {path} must contain at least one tensor among logits/prob/pred."
        )
    row_count = 0
    for value in (prob, logits, pred):
        if value is not None:
            row_count = int(value.shape[0])
            break
    max_target = max([int(item) for item in target_node_ids], default=-1)
    if max_target >= row_count:
        raise ValueError(
            f"Base outputs at {path} have {row_count} rows but routed target node {max_target} is requested."
        )
    return {
        "path": str(path),
        "keys": sorted(str(key) for key in raw.keys()),
        "logits": logits,
        "prob": prob,
        "pred": pred,
        "row_count": int(row_count),
    }


def _residual_base_state_for_node(base_outputs, node_id, class_names):
    if base_outputs is None:
        return None
    node_id = int(node_id)
    prob = base_outputs.get("prob")
    logits = base_outputs.get("logits")
    pred = base_outputs.get("pred")
    prob_values = None
    if prob is not None:
        prob_values = [float(item) for item in prob[node_id].float().view(-1).tolist()]
    elif logits is not None:
        prob_values = [float(item) for item in torch.softmax(logits[node_id].float().view(-1), dim=-1).tolist()]
    if pred is not None:
        pred_idx = int(pred[node_id].view(-1)[0].item())
    elif prob_values:
        pred_idx = int(np.argmax(prob_values))
    else:
        pred_idx = -1
    if 0 <= pred_idx < len(class_names):
        pred_label = str(class_names[pred_idx])
    else:
        pred_label = f"class_{pred_idx}" if pred_idx >= 0 else "unknown"
    confidence = max(prob_values) if prob_values else None
    if prob_values and len(prob_values) >= 2:
        sorted_probs = sorted(prob_values, reverse=True)
        margin = float(sorted_probs[0] - sorted_probs[1])
    else:
        margin = None
    return {
        "pred_idx": int(pred_idx),
        "pred_label": pred_label,
        "prob_values": prob_values,
        "confidence": confidence,
        "margin": margin,
    }


def _format_residual_base_state(base_state, class_names, variant):
    if base_state is None:
        return "BASE_MODEL_STATE:\n- omitted for no_base variant"
    lines = ["BASE_MODEL_STATE:"]
    if variant == "base_as_assertion":
        lines.append(
            f"- frozen SimTeG prediction: {base_state['pred_label']} (strong prior for anchoring negative-control)"
        )
    else:
        lines.append(
            f"- frozen SimTeG prediction: {base_state['pred_label']} (fallible prior, not ground truth)"
        )
    if base_state.get("confidence") is not None:
        lines.append(f"- confidence: {float(base_state['confidence']):.4f}")
    if base_state.get("margin") is not None:
        lines.append(f"- probability margin between top two classes: {float(base_state['margin']):.4f}")
    prob_values = base_state.get("prob_values")
    if prob_values:
        for idx, value in enumerate(prob_values):
            label = str(class_names[idx]) if idx < len(class_names) else f"class_{idx}"
            lines.append(f"- probability_{label}: {float(value):.4f}")
    return "\n".join(lines)


def _residual_profile_cues(record):
    return "\n".join(
        [
            f"- display_name: {record['display_name'] or 'Unknown'}",
            f"- handle: @{record['screen_name']}" if record["screen_name"] else "- handle: unknown",
            f"- verified: {_format_bool(record['verified'])}",
            f"- protected: {_format_bool(record['protected'])}",
            f"- bio_present: {_format_bool(record['bio_present'])}",
            f"- account_age_bucket: {record['account_age_bucket']}",
            f"- follow_ratio_bucket: {record['follow_ratio_bucket']}",
            f"- posting_density_bucket: {record['posting_density_bucket']}",
            f"- followers_count: {record['followers_count']}",
            f"- following_count: {record['following_count']}",
            f"- listed_count: {record['listed_count']}",
            f"- status_count: {record['statuses_count']}",
            f"- bio: {_truncate_chars(record['bio'], 180) or 'None'}",
        ]
    )


def _residual_tweet_evidence(record, tweet_stats):
    lines = [
        f"- total_non_empty_tweets: {int(tweet_stats['tweet_count'])}",
        f"- sampled_tweets: {int(len(tweet_stats['sampled_tweets']))}",
        f"- retweet_ratio: {float(tweet_stats['rt_ratio']):.2f}",
        f"- url_ratio: {float(tweet_stats['url_ratio']):.2f}",
        f"- hashtag_ratio: {float(tweet_stats['hashtag_ratio']):.2f}",
        f"- duplicate_ratio: {float(tweet_stats['duplicate_ratio']):.2f}",
        f"- average_tweet_length_bucket: {tweet_stats['avg_tweet_len_bucket']}",
    ]
    if tweet_stats["sampled_tweets"]:
        lines.append("- representative_tweets:")
        for tweet in tweet_stats["sampled_tweets"][:6]:
            lines.append(f"  - {_truncate_chars(tweet, 220)}")
    else:
        lines.append("- representative_tweets: None")
    return "\n".join(lines)


def _base_prediction_distribution(node_ids, base_outputs, class_names):
    if base_outputs is None:
        return "not_available"
    counts = {str(label): 0 for label in class_names}
    unknown = 0
    pred = base_outputs.get("pred")
    if pred is None:
        return "not_available"
    row_count = int(pred.shape[0])
    for node_id in sorted(set(int(item) for item in node_ids if int(item) >= 0)):
        if int(node_id) >= row_count:
            unknown += 1
            continue
        idx = int(pred[int(node_id)].view(-1)[0].item())
        label = str(class_names[idx]) if 0 <= idx < len(class_names) else "unknown"
        if label in counts:
            counts[label] += 1
        else:
            unknown += 1
    parts = [f"{label}={count}" for label, count in counts.items()]
    if unknown:
        parts.append(f"unknown={unknown}")
    return ", ".join(parts)


def _residual_graph_context(node_id, context, base_outputs, class_names, include_neighbor_base_distribution):
    following = sorted(set(int(item) for item in context["following"][int(node_id)] if int(item) != int(node_id)))
    follower = sorted(set(int(item) for item in context["follower"][int(node_id)] if int(item) != int(node_id)))
    undirected = sorted(set(int(item) for item in context["undirected"][int(node_id)] if int(item) != int(node_id)))
    reciprocal = sorted(set(following).intersection(set(follower)))
    lines = [
        f"- following_count: {len(following)}",
        f"- follower_count: {len(follower)}",
        f"- one_hop_undirected_count: {len(undirected)}",
        f"- reciprocal_follow_count: {len(reciprocal)}",
        f"- has_following: {_format_bool(bool(following))}",
        f"- has_follower: {_format_bool(bool(follower))}",
    ]
    if include_neighbor_base_distribution and base_outputs is not None:
        lines.extend(
            [
                f"- following_neighbor_base_prediction_distribution: {_base_prediction_distribution(following, base_outputs, class_names)}",
                f"- follower_neighbor_base_prediction_distribution: {_base_prediction_distribution(follower, base_outputs, class_names)}",
                f"- one_hop_neighbor_base_prediction_distribution: {_base_prediction_distribution(undirected, base_outputs, class_names)}",
            ]
        )
    return "\n".join(lines)


def _residual_task_block(variant):
    if variant == "no_base":
        return "\n".join(
            [
                "<task>",
                "Assess this routed Twitter account using only observable profile, tweet, and local graph evidence.",
                "The frozen detector prediction is intentionally hidden in this ablation.",
                "Return an evidence-grounded bot/human/uncertain judgment without relying on base-model state.",
                "</task>",
            ]
        )
    if variant == "base_as_assertion":
        return "\n".join(
            [
                "<task>",
                "Audit a frozen SimTeG Twitter bot detector prediction for one routed account.",
                "Treat the base prediction as a strong prior. Revise it only when the observable evidence clearly contradicts it.",
                "This is a negative-control prompt for measuring anchoring behavior.",
                "</task>",
            ]
        )
    return "\n".join(
        [
            "<task>",
            "Audit a frozen SimTeG Twitter bot detector prediction for one routed account.",
            "Treat the base prediction as a fallible prior, not ground truth.",
            "First assess the observable evidence independently; then decide whether it justifies changing the base prediction.",
            "</task>",
        ]
    )


def _residual_system_instruction(variant):
    base_policy = (
        "The base-model prediction is hidden; judge only from observable evidence."
        if variant == "no_base"
        else "The base-model prediction is a prior, not a label. Do not agree with it unless evidence supports it."
        if variant == "base_as_hypothesis"
        else "The base-model prediction is a strong prior in this negative-control variant, but observable contradictions still matter."
    )
    return "\n".join(
        [
            "You are a cautious evidence auditor for Twitter social-bot detection.",
            "Use only the supplied profile, tweet, and local graph evidence.",
            base_policy,
            "Distinguish evidence for bot-like automation from benign explanations such as fandom, news sharing, professional branding, or sparse data.",
            "Do not infer from dataset labels, oracle outcomes, or unavailable information.",
            "Return only one valid JSON object. Do not include markdown, prose outside JSON, or hidden chain-of-thought.",
        ]
    )


def _residual_decision_policy(variant):
    if variant == "no_base":
        return "\n".join(
            [
                "<decision_policy>",
                "1. Decide an evidence_label from observable evidence: human, bot, or uncertain.",
                "2. Use uncertain when evidence is sparse, mixed, or mostly explained by benign behavior.",
                "3. Keep evidence arrays short and source-grounded.",
                "</decision_policy>",
            ]
        )
    return "\n".join(
        [
            "<decision_policy>",
            "1. Decide evidence_label from observable evidence before using the base-model state: human, bot, or uncertain.",
            "2. Compare evidence_label with the base prediction.",
            "3. Choose keep_base when evidence is uncertain, sparse, mixed, or only weakly contradicts the base prediction.",
            "4. Choose change_to_human or change_to_bot only when specific counter-evidence is stronger than the risk of breaking a correct base prediction.",
            "5. Dense-neighborhood or neighbor-prediction disagreement is a risk cue, not proof by itself.",
            "</decision_policy>",
        ]
    )


def _residual_output_schema(variant):
    if variant == "no_base":
        action_line = '"prediction": "human | bot | uncertain",'
    else:
        action_line = '"correction_action": "keep_base | change_to_human | change_to_bot",'
    return "\n".join(
        [
            "<output_json_schema>",
            "{",
            '  "evidence_label": "human | bot | uncertain",',
            f"  {action_line}",
            '  "confidence": "low | medium | high",',
            '  "bot_like_evidence": ["short source-grounded cue", "..."],',
            '  "human_or_benign_evidence": ["short source-grounded cue", "..."],',
            '  "uncertainty_or_risk": ["short source-grounded cue", "..."]',
            "}",
            "</output_json_schema>",
        ]
    )


def _build_residual_audit_prompt_parts(record, tweet_stats, graph_context_text, base_state, class_names, variant):
    label_space = "\n".join([f"- {label}" for label in class_names])
    system = _residual_system_instruction(variant)
    evidence_sections = [
        "<observable_evidence>",
        "<node_profile_cues>\n" + _residual_profile_cues(record) + "\n</node_profile_cues>",
        "<tweet_evidence>\n" + _residual_tweet_evidence(record, tweet_stats) + "\n</tweet_evidence>",
        "<local_graph_context>\n" + graph_context_text + "\n</local_graph_context>",
        "</observable_evidence>",
    ]
    base_section = "<base_model_state>\n" + _format_residual_base_state(base_state, class_names, variant) + "\n</base_model_state>"
    user_sections = [
        _residual_task_block(variant),
        "<label_space>\n" + label_space + "\n</label_space>",
    ]
    if variant == "base_as_assertion":
        user_sections.append(base_section)
    user_sections.extend(evidence_sections)
    if variant == "base_as_hypothesis":
        user_sections.append(base_section)
    user_sections.extend([_residual_decision_policy(variant), _residual_output_schema(variant)])
    user = "\n\n".join(user_sections)
    full_prompt = "\n\n".join(
        [
            "SYSTEM_MESSAGE:",
            system,
            "USER_MESSAGE:",
            user,
            "ASSISTANT_JSON:",
        ]
    )
    return {"system": system, "user": user, "full": full_prompt}


def _build_residual_audit_prompt(record, tweet_stats, graph_context_text, base_state, class_names, variant):
    return _build_residual_audit_prompt_parts(
        record,
        tweet_stats,
        graph_context_text,
        base_state,
        class_names,
        variant,
    )["full"]


def _resolve_residual_audit_prompt_bundle(
    args,
    records,
    edge_index,
    edge_type,
    classes,
    target_node_ids=None,
    base_outputs=None,
):
    num_nodes = len(records)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    class_names = _residual_class_names(classes)
    variant = str(getattr(args, "residual_prompt_variant", "base_as_hypothesis") or "base_as_hypothesis").strip().lower()
    if variant not in RESIDUAL_PROMPT_VARIANT_CHOICES:
        raise ValueError(f"Unsupported residual_prompt_variant: {variant}")
    if variant != "no_base" and base_outputs is None:
        raise ValueError(f"--residual_base_outputs_path is required for residual_prompt_variant={variant}.")
    include_neighbor_base_distribution = bool(getattr(args, "residual_include_neighbor_base_distribution", True))
    if variant == "no_base":
        include_neighbor_base_distribution = False
    prompts = []
    prompt_rows = []
    counts = {
        "residual_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "tweet_nodes": [],
    }
    for node_id in target_ids:
        record = records[int(node_id)]
        tweet_stats = _get_tweet_stats(record, sample_size=int(getattr(args, "tweet_sample_size", 6)))
        base_state = _residual_base_state_for_node(base_outputs, node_id, class_names) if base_outputs is not None else None
        graph_context_text = _residual_graph_context(
            node_id,
            context,
            base_outputs,
            class_names,
            include_neighbor_base_distribution=include_neighbor_base_distribution,
        )
        prompt_parts = _build_residual_audit_prompt_parts(
            record,
            tweet_stats,
            graph_context_text,
            base_state,
            class_names,
            variant,
        )
        prompt = prompt_parts["full"]
        prompts.append(prompt)
        counts["residual_nodes"].append(1)
        counts["following_nodes"].append(len(set(context["following"][int(node_id)])))
        counts["follower_nodes"].append(len(set(context["follower"][int(node_id)])))
        counts["tweet_nodes"].append(len(tweet_stats.get("sampled_tweets", [])))
        prompt_rows.append(
            {
                "node_id": int(node_id),
                "component_name": "residual_audit",
                "prompt_role": "base_prediction_residual_auditor",
                "residual_prompt_variant": variant,
                "base_prediction": base_state.get("pred_label") if base_state else "",
                "base_confidence": float(base_state["confidence"]) if base_state and base_state.get("confidence") is not None else None,
                "base_margin": float(base_state["margin"]) if base_state and base_state.get("margin") is not None else None,
                "system": prompt_parts["system"],
                "user": prompt_parts["user"],
                "prompt": prompt,
            }
        )
    return {
        "prompt_family": "residual_audit_v1",
        "prompt_style": f"residual_audit_v1_{variant}",
        "prompt_family_version": "residual_audit_v1",
        "residual_prompt_variant": variant,
        "evidence_schema": "residual_audit",
        "evidence_card_fields": [
            "evidence_label",
            "prediction" if variant == "no_base" else "correction_action",
            "confidence",
            "bot_like_evidence",
            "human_or_benign_evidence",
            "uncertainty_or_risk",
        ],
        "prompt_components": {"residual_audit": prompts},
        "structured_components": {},
        "structured_component_schema": {},
        "generation_rows_by_component": {},
        "component_max_length_group": {"residual_audit": "hop"},
        "component_prompt_roles": {"residual_audit": "base_prediction_residual_auditor"},
        "counts": counts,
        "neighbor_sample_policy": "base_prediction_residual_audit_context",
        "component_order": {"residual_audit_v1": ["residual_audit"]},
        "selected_components": ["residual_audit"],
        "scalar_features": {},
        "prompt_rows": prompt_rows,
        "semantic_view_mode": "residual_audit_v1",
        "selection_policy_note": (
            "Residual-audit prompts expose SimTeG base predictions as fallible hypotheses and append compact "
            "profile, tweet, and local graph evidence; true labels are never inserted into the prompt."
        ),
        "needs_conflict_generation": False,
        "base_outputs_path": str((base_outputs or {}).get("path", "")),
        "base_outputs_row_count": int((base_outputs or {}).get("row_count", 0) or 0),
        "base_output_keys": list((base_outputs or {}).get("keys", [])),
        "include_neighbor_base_distribution": bool(include_neighbor_base_distribution),
        "class_names": class_names,
    }


def _dgp_system_instruction():
    return "You are a social bot detection classifier."


def _dgp_target_block(record, tweet_stats):
    tweets = list(tweet_stats.get("sampled_tweets", []) or [])
    tweet_lines = [f"- {tweet}" for tweet in tweets] if tweets else ["- None"]
    return "\n".join(
        [
            "EGO_PROFILE:",
            _profile_card(record, brief=False),
            "EGO_TWEET_BEHAVIOR:",
            _tweet_behavior_summary(record, tweet_stats),
            "EGO_TWEET_SAMPLES:",
            *tweet_lines,
        ]
    )


def _dgp_neighbor_card(record):
    return (
        f"- {record['display_name'] or 'Unknown'} (@{record['screen_name'] or 'unknown'}) | "
        f"verified={_format_bool(record['verified'])}, protected={_format_bool(record['protected'])}, "
        f"bio_present={_format_bool(record['bio_present'])}, age_bucket={record['account_age_bucket']}, "
        f"follow_ratio_bucket={record['follow_ratio_bucket']}, posting_density_bucket={record['posting_density_bucket']}, "
        f"followers={int(record['followers_count'])}, following={int(record['following_count'])}, "
        f"bio={_truncate_chars(record['bio'], 80) or 'None'}"
    )


def _dgp_select_neighbors(node_id, context, records, direction, quota):
    candidates = list(context[direction][int(node_id)])
    record_texts = [_compact_whitespace(record.get("bio", "")) for record in records]
    ranked = _rank_directional_candidates(int(node_id), candidates, record_texts, context)
    return ranked[: int(quota)]


def _dgp_neighbor_context(node_id, context, records, args, variant):
    following = sorted(set(int(item) for item in context["following"][int(node_id)] if int(item) != int(node_id)))
    follower = sorted(set(int(item) for item in context["follower"][int(node_id)] if int(item) != int(node_id)))
    reciprocal = sorted(set(following).intersection(set(follower)))
    lines = [
        "GRAPH_CONTEXT_STATS:",
        f"- following_count={len(following)}",
        f"- follower_count={len(follower)}",
        f"- reciprocal_follow_count={len(reciprocal)}",
        f"- has_following={_format_bool(bool(following))}",
        f"- has_follower={_format_bool(bool(follower))}",
    ]
    if str(variant) == "target_only":
        lines.extend(
            [
                "- neighbor_cards_policy=omitted_target_only_ablation",
            ]
        )
        return "\n".join(lines), {"following_selected": 0, "follower_selected": 0}

    following_selected = _dgp_select_neighbors(
        node_id,
        context,
        records,
        "following",
        int(getattr(args, "following_quota", 3)),
    )
    follower_selected = _dgp_select_neighbors(
        node_id,
        context,
        records,
        "follower",
        int(getattr(args, "follower_quota", 3)),
    )
    lines.extend(
        [
            "FOLLOWING_NEIGHBORS:",
            *([_dgp_neighbor_card(records[int(item)]) for item in following_selected] or ["- None"]),
            "FOLLOWER_NEIGHBORS:",
            *([_dgp_neighbor_card(records[int(item)]) for item in follower_selected] or ["- None"]),
        ]
    )
    return "\n".join(lines), {
        "following_selected": int(len(following_selected)),
        "follower_selected": int(len(follower_selected)),
    }


def _build_dgp_predictor_prompt_parts(record, tweet_stats, neighbor_context_text, class_names, variant):
    system = _dgp_system_instruction()
    label_space = ", ".join(str(label) for label in class_names)
    user = "\n\n".join(
        [
            f"Instruct: Predict the node's category for social bot detection from the provided context. Possible categories: {label_space}.",
            "Query:",
            _dgp_target_block(record, tweet_stats),
            neighbor_context_text,
            "Answer with exactly one token: Yes or No.",
        ]
    )
    full_prompt = "\n\n".join(
        [
            "SYSTEM_MESSAGE:",
            system,
            "USER_MESSAGE:",
            user,
            "ASSISTANT_ANSWER:",
        ]
    )
    return {"system": system, "user": user, "full": full_prompt}


def _dgp_v2_direction_components(variant):
    variant = str(variant or "norm_text_following_summary").strip().lower()
    if variant == "norm_text_following_summary":
        return ("following",)
    if variant == "norm_text_follower_summary":
        return ("follower",)
    if variant == "norm_text_following_follower_summary":
        return ("following", "follower")
    if variant == "norm_text_target_only":
        return tuple()
    raise ValueError(f"Unsupported dgp_predictor_v2 prompt variant: {variant}")


def _dgp_v2_relation_label(direction):
    return "following" if str(direction) == "following" else "follower"


def _dgp_v2_relation_description(direction):
    if str(direction) == "following":
        return "accounts the target user chooses to follow"
    return "accounts that follow the target user"


def _dgp_v2_clean_evidence_text(text, limit=480):
    text = html.unescape(str(text or ""))
    text = re.sub(r"</?s>", " ", text)
    text = re.sub(r"<\|[^>]+>", " ", text)
    text = re.sub(r"</?[A-Za-z_][A-Za-z0-9_:-]*>", " ", text)
    text = re.sub(r"@USER\b", "user mention", text)
    text = re.sub(r"HTTPURL\b", "URL", text)
    text = re.sub(r"#HASHTAG\b", "hashtag", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069]", " ", text)
    text = _compact_whitespace(text)
    return _truncate_chars(text, int(limit))


def _dgp_v2_render_norm_user_text(raw_text, limit=2400):
    raw_text = str(raw_text or "")
    if "METADATA:" not in raw_text and "DESCRIPTION:" not in raw_text and "TWEET:" not in raw_text:
        return _dgp_v2_clean_evidence_text(raw_text, limit=limit) or "No usable account text was available."
    record = _parse_norm_user_record(raw_text, node_id=-1)
    record = dict(record)
    record["bio"] = _dgp_v2_clean_evidence_text(record.get("bio", ""), limit=240)
    record["tweets"] = [
        cleaned
        for cleaned in (
            _dgp_v2_clean_evidence_text(tweet, limit=260)
            for tweet in record.get("tweets", [])
        )
        if cleaned
    ]
    tweet_stats = _get_tweet_stats(record, sample_size=5)
    lines = [
        "PROFILE:",
        _profile_cue_summary(record, include_identity=True),
        "TWEET_BEHAVIOR:",
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
    ]
    rendered = "\n".join(lines)
    return _truncate_chars(rendered, int(limit)) or "No usable account text was available."


def _dgp_v2_clean_norm_text(text, limit=2400):
    return _dgp_v2_render_norm_user_text(text, limit=limit)


def _dgp_v2_clean_generated_summary(text, limit=600):
    text = _sanitize_generated_explanation(text)
    text = _dgp_v2_clean_evidence_text(text, limit=limit)
    return text or "No usable summary was available."


def _dgp_v2_summary_system():
    return "You summarize account text."


def _dgp_v2_neighbor_summary_prompt(direction, neighbor_text):
    direction = _dgp_v2_relation_label(direction)
    user = "\n\n".join(
        [
            "Instruct: Summarize the following account text within 10 tokens.",
            "Query:",
            f"RELATION_TO_TARGET: {direction}",
            "ACCOUNT_TEXT:",
            _dgp_v2_clean_norm_text(neighbor_text, limit=2200),
            "Summary:",
        ]
    )
    return {
        "system": _dgp_v2_summary_system(),
        "user": user,
        "fallback_explanation": (
            f"The selected {direction} neighbor provides limited account text. "
            f"{_dgp_v2_clean_norm_text(neighbor_text, limit=360)}"
        ),
        "prompt_role": f"dgp_v2_{direction}_neighbor_summary",
    }


def _dgp_v2_context_summary_prompt(direction, neighbor_summary_rows):
    direction = _dgp_v2_relation_label(direction)
    relation_description = _dgp_v2_relation_description(direction)
    if neighbor_summary_rows:
        summary_lines = [
            f"{idx + 1}. {str(row.get('summary', '') or 'No summary available.').strip()}"
            for idx, row in enumerate(neighbor_summary_rows)
        ]
    else:
        summary_lines = ["None"]
    user = "\n\n".join(
        [
            "Instruct: Summarize the following account texts within 10 tokens.",
            "Query:",
            f"RELATION_CONTEXT: {direction}",
            f"RELATION_DESCRIPTION: {relation_description}",
            "ACCOUNT_SUMMARIES:",
            "\n".join(summary_lines),
            "Summary:",
        ]
    )
    return {
        "system": _dgp_v2_summary_system(),
        "user": user,
        "fallback_explanation": (
            f"The {direction} context is weak or unavailable."
            if not neighbor_summary_rows
            else " ".join(str(row.get("summary", "") or "") for row in neighbor_summary_rows)[:600]
        ),
        "prompt_role": f"dgp_v2_{direction}_context_summary",
    }


def _dgp_v2_empty_context_summary(direction):
    direction = _dgp_v2_relation_label(direction)
    return (
        f"The {direction} context is weak or unavailable because no selected "
        f"{direction} neighbors were available for this target account."
    )


def _dgp_v2_predictor_system():
    return "You are a social bot detection classifier."


def _dgp_v2_predictor_prompt_parts(target_text, context_summaries, variant):
    context_sections = []
    for direction in ("following", "follower"):
        summary = str(context_summaries.get(direction, "") or "").strip()
        if not summary:
            continue
        tag = f"{direction.upper()}_CONTEXT_SUMMARY"
        context_sections.extend([f"{tag}:", summary])
    if not context_sections:
        context_sections = [
            "NEIGHBOR_CONTEXT_SUMMARY:",
            "No neighbor context summary is used for this target-only ablation.",
        ]
    user = "\n\n".join(
        [
            "Instruct: Predict the node's category for social bot detection from the provided context. Possible categories: No, Yes.",
            "Query:",
            "TARGET_ACCOUNT_TEXT:",
            _dgp_v2_clean_norm_text(target_text, limit=3600),
            "\n".join(context_sections),
            "Answer with exactly one token: Yes or No.",
        ]
    )
    full_prompt = "\n\n".join(
        [
            "SYSTEM_MESSAGE:",
            _dgp_v2_predictor_system(),
            "USER_MESSAGE:",
            user,
            "ASSISTANT_ANSWER:",
        ]
    )
    return {"system": _dgp_v2_predictor_system(), "user": user, "full": full_prompt}


def _resolve_dgp_predictor_v2_prompt_bundle(
    args,
    texts,
    edge_index,
    edge_type,
    classes,
    target_node_ids=None,
):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    variant = str(getattr(args, "dgp_prompt_variant", "norm_text_following_summary") or "").strip().lower()
    if variant in {"target_fine_neighbor_coarse", "target_only"}:
        variant = "norm_text_following_summary"
    directions = _dgp_v2_direction_components(variant)
    dgp_k = int(getattr(args, "dgp_neighbor_summary_k", 5))
    following_quota = dgp_k
    follower_quota = dgp_k
    if following_quota <= 0 or follower_quota <= 0:
        raise ValueError("dgp_predictor_v2 requires a positive --dgp_neighbor_summary_k value.")
    prompts = []
    prompt_rows = []
    generation_rows_by_component = {"dgp_neighbor_summary": [], "dgp_context_summary": []}
    counts = {
        "dgp_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "selected_following_nodes": [],
        "selected_follower_nodes": [],
    }
    records_text = [_compact_whitespace(text) for text in texts]
    for node_id in target_ids:
        node_id = int(node_id)
        following = sorted(set(int(item) for item in context["following"][node_id] if int(item) != node_id))
        follower = sorted(set(int(item) for item in context["follower"][node_id] if int(item) != node_id))
        selected_by_direction = {
            "following": _rank_directional_candidates(node_id, following, records_text, context)[:following_quota]
            if "following" in directions
            else [],
            "follower": _rank_directional_candidates(node_id, follower, records_text, context)[:follower_quota]
            if "follower" in directions
            else [],
        }
        row = {
            "node_id": node_id,
            "component_name": "dgp_predictor",
            "prompt_role": "dgp_v2_norm_text_neighbor_summary_predictor",
            "dgp_prompt_variant": variant,
            "target_norm_user_text": _dgp_v2_clean_norm_text(texts[node_id], limit=3600),
            "selected_components": ["dgp_predictor"],
            "selected_following_nodes": [int(item) for item in selected_by_direction["following"]],
            "selected_follower_nodes": [int(item) for item in selected_by_direction["follower"]],
            "following_context_summary": "",
            "follower_context_summary": "",
            "neighbor_summary_nodes": [],
        }
        for direction in directions:
            for rank, neighbor_id in enumerate(selected_by_direction[direction], start=1):
                generation_prompt = _dgp_v2_neighbor_summary_prompt(direction, texts[int(neighbor_id)])
                generation_rows_by_component["dgp_neighbor_summary"].append(
                    {
                        "node_id": node_id,
                        "component_name": "dgp_neighbor_summary",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": generation_prompt["prompt_role"],
                        "summary_direction": direction,
                        "neighbor_node_id": int(neighbor_id),
                        "neighbor_rank": int(rank),
                    }
                )
                row["neighbor_summary_nodes"].append(
                    {
                        "direction": direction,
                        "neighbor_node_id": int(neighbor_id),
                        "neighbor_rank": int(rank),
                    }
                )
        counts["dgp_nodes"].append(1)
        counts["following_nodes"].append(len(following))
        counts["follower_nodes"].append(len(follower))
        counts["selected_following_nodes"].append(len(selected_by_direction["following"]))
        counts["selected_follower_nodes"].append(len(selected_by_direction["follower"]))
        prompts.append("")
        prompt_rows.append(row)
    if not directions:
        generation_rows_by_component = {}
        for idx, row in enumerate(prompt_rows):
            prompt_parts = _dgp_v2_predictor_prompt_parts(row["target_norm_user_text"], {}, variant)
            row.update(
                {
                    "system": prompt_parts["system"],
                    "user": prompt_parts["user"],
                    "prompt": prompt_parts["full"],
                }
            )
            prompts[idx] = prompt_parts["full"]
    return {
        "prompt_family": "dgp_predictor_v2",
        "prompt_style": f"dgp_predictor_v2_{variant}",
        "prompt_family_version": "dgp_predictor_v2",
        "dgp_prompt_variant": variant,
        "dgp_neighbor_summary_k": {"following": following_quota, "follower": follower_quota},
        "dgp_v2_directions": list(directions),
        "evidence_schema": "dgp_norm_text_neighbor_summary_predictor",
        "norm_user_text_rendering": "llm_friendly_profile_tweet_behavior_samples",
        "norm_user_text_rendering_note": (
            "Raw serialized norm_user_text is parsed and rendered into PROFILE, "
            "TWEET_BEHAVIOR, and TWEET_SAMPLES sections before entering Qwen; "
            "special tokens, XML-like tags, @USER placeholders, and HTTPURL markers are cleaned."
        ),
        "dgp_v2_summary_language": "english_only",
        "dgp_v2_empty_context_policy": "deterministic_sparse_summary_no_llm",
        "evidence_card_fields": ["answer_token"],
        "prompt_components": {"dgp_predictor": prompts},
        "structured_components": {},
        "structured_component_schema": {},
        "generation_rows_by_component": generation_rows_by_component,
        "component_max_length_group": {
            "dgp_predictor": "hop",
            "dgp_neighbor_summary": "hop",
            "dgp_context_summary": "hop",
        },
        "component_prompt_roles": {
            "dgp_predictor": "dgp_v2_norm_text_neighbor_summary_yes_no_predictor",
            "dgp_neighbor_summary": "dgp_v2_selected_neighbor_norm_text_summary",
            "dgp_context_summary": "dgp_v2_relation_context_summary",
        },
        "counts": counts,
        "neighbor_sample_policy": "dgp_v2_directional_topk_norm_user_text_summary",
        "component_order": {"dgp_predictor_v2": ["dgp_predictor"]},
        "selected_components": ["dgp_predictor"],
        "scalar_features": {},
        "prompt_rows": prompt_rows,
        "semantic_view_mode": "dgp_predictor_v2",
        "selection_policy_note": (
            "DGP v2 uses target norm_user_text as fine-grained tweet+metadata evidence, summarizes selected "
            "K=5 directional neighbor norm_user_text rows, then predicts Yes/No from target text plus coarse "
            "relation summaries. True labels, base predictions, and oracle correction outcomes are never inserted."
        ),
        "class_names": ["No", "Yes"],
        "needs_dgp_v2_context_generation": bool(directions),
    }


def _attach_dgp_v2_neighbor_summaries(prompt_bundle, explanations):
    generation_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get("dgp_neighbor_summary", []))
    if generation_rows and len(explanations) != len(generation_rows):
        raise ValueError("Generated DGP v2 neighbor summary count does not match generation rows.")
    row_by_node = {int(row["node_id"]): row for row in prompt_bundle.get("prompt_rows", [])}
    summaries_by_node_direction = defaultdict(list)
    for generation_row, explanation in zip(generation_rows, explanations):
        node_id = int(generation_row["node_id"])
        direction = _dgp_v2_relation_label(generation_row.get("summary_direction", "following"))
        item = {
            "direction": direction,
            "neighbor_node_id": int(generation_row.get("neighbor_node_id", -1)),
            "neighbor_rank": int(generation_row.get("neighbor_rank", 0)),
            "summary": str(explanation or "").strip(),
        }
        summaries_by_node_direction[(node_id, direction)].append(item)
    context_generation_rows = []
    for row in prompt_bundle.get("prompt_rows", []):
        node_id = int(row["node_id"])
        row["dgp_v2_neighbor_summaries"] = []
        for direction in prompt_bundle.get("dgp_v2_directions", []):
            summaries = sorted(
                summaries_by_node_direction.get((node_id, direction), []),
                key=lambda item: int(item.get("neighbor_rank", 0)),
            )
            row[f"{direction}_neighbor_summaries"] = summaries
            row["dgp_v2_neighbor_summaries"].extend(summaries)
            if not summaries:
                row[f"{direction}_context_summary"] = _dgp_v2_empty_context_summary(direction)
                continue
            generation_prompt = _dgp_v2_context_summary_prompt(direction, summaries)
            context_generation_rows.append(
                {
                    "node_id": node_id,
                    "component_name": "dgp_context_summary",
                    "system": generation_prompt["system"],
                    "user": generation_prompt["user"],
                    "fallback_explanation": generation_prompt["fallback_explanation"],
                    "prompt_role": generation_prompt["prompt_role"],
                    "summary_direction": direction,
                }
            )
    prompt_bundle.setdefault("generation_rows_by_component", {})["dgp_context_summary"] = context_generation_rows
    return prompt_bundle


def _attach_dgp_v2_context_summaries(prompt_bundle, explanations):
    generation_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get("dgp_context_summary", []))
    if generation_rows and len(explanations) != len(generation_rows):
        raise ValueError("Generated DGP v2 context summary count does not match generation rows.")
    context_by_node = defaultdict(dict)
    for generation_row, explanation in zip(generation_rows, explanations):
        node_id = int(generation_row["node_id"])
        direction = _dgp_v2_relation_label(generation_row.get("summary_direction", "following"))
        context_by_node[node_id][direction] = str(explanation or "").strip()
    prompts = []
    variant = str(prompt_bundle.get("dgp_prompt_variant", "norm_text_following_summary") or "norm_text_following_summary")
    for row in prompt_bundle.get("prompt_rows", []):
        node_id = int(row["node_id"])
        context_summaries = {
            direction: str(row.get(f"{direction}_context_summary", "") or "").strip()
            for direction in prompt_bundle.get("dgp_v2_directions", [])
            if str(row.get(f"{direction}_context_summary", "") or "").strip()
        }
        context_summaries.update(dict(context_by_node.get(node_id, {})))
        for direction, summary in context_summaries.items():
            row[f"{direction}_context_summary"] = summary
        prompt_parts = _dgp_v2_predictor_prompt_parts(row.get("target_norm_user_text", ""), context_summaries, variant)
        row.update(
            {
                "system": prompt_parts["system"],
                "user": prompt_parts["user"],
                "prompt": prompt_parts["full"],
            }
        )
        prompts.append(prompt_parts["full"])
    prompt_bundle["prompt_components"]["dgp_predictor"] = prompts
    return prompt_bundle


def _resolve_dgp_predictor_prompt_bundle(
    args,
    records,
    edge_index,
    edge_type,
    classes,
    target_node_ids=None,
):
    num_nodes = len(records)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    class_names = _residual_class_names(classes)
    variant = str(getattr(args, "dgp_prompt_variant", "target_fine_neighbor_coarse") or "").strip().lower()
    if variant not in {"target_fine_neighbor_coarse", "target_only"}:
        raise ValueError(f"Unsupported dgp_prompt_variant: {variant}")
    prompts = []
    prompt_rows = []
    counts = {
        "dgp_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "tweet_nodes": [],
        "selected_following_nodes": [],
        "selected_follower_nodes": [],
    }
    for node_id in target_ids:
        record = records[int(node_id)]
        tweet_stats = _get_tweet_stats(record, sample_size=int(getattr(args, "tweet_sample_size", 6)))
        neighbor_context_text, neighbor_stats = _dgp_neighbor_context(node_id, context, records, args, variant)
        prompt_parts = _build_dgp_predictor_prompt_parts(
            record,
            tweet_stats,
            neighbor_context_text,
            class_names,
            variant,
        )
        prompt = prompt_parts["full"]
        prompts.append(prompt)
        following_count = len(set(context["following"][int(node_id)]))
        follower_count = len(set(context["follower"][int(node_id)]))
        counts["dgp_nodes"].append(1)
        counts["following_nodes"].append(following_count)
        counts["follower_nodes"].append(follower_count)
        counts["tweet_nodes"].append(len(tweet_stats.get("sampled_tweets", [])))
        counts["selected_following_nodes"].append(int(neighbor_stats["following_selected"]))
        counts["selected_follower_nodes"].append(int(neighbor_stats["follower_selected"]))
        prompt_rows.append(
            {
                "node_id": int(node_id),
                "component_name": "dgp_predictor",
                "prompt_role": "dgp_dual_granularity_bot_human_predictor",
                "dgp_prompt_variant": variant,
                "system": prompt_parts["system"],
                "user": prompt_parts["user"],
                "prompt": prompt,
            }
        )
    return {
        "prompt_family": "dgp_predictor_v1",
        "prompt_style": f"dgp_predictor_v1_{variant}",
        "prompt_family_version": "dgp_predictor_v1",
        "dgp_prompt_variant": variant,
        "evidence_schema": "dgp_dual_granularity_predictor",
        "evidence_card_fields": ["label"],
        "prompt_components": {"dgp_predictor": prompts},
        "structured_components": {},
        "structured_component_schema": {},
        "generation_rows_by_component": {},
        "component_max_length_group": {"dgp_predictor": "hop"},
        "component_prompt_roles": {"dgp_predictor": "dgp_dual_granularity_bot_human_predictor"},
        "counts": counts,
        "neighbor_sample_policy": "dgp_target_fine_neighbor_coarse_ranked_neighbors",
        "component_order": {"dgp_predictor_v1": ["dgp_predictor"]},
        "selected_components": ["dgp_predictor"],
        "scalar_features": {},
        "prompt_rows": prompt_rows,
        "semantic_view_mode": "dgp_predictor_v1",
        "selection_policy_note": (
            "DGP-style prompts keep detailed target profile/tweet evidence and compress following/follower "
            "neighbors into coarse ranked cards. True labels and oracle correction outcomes are never inserted."
        ),
        "class_names": class_names,
    }


def _expert_output_name(prompt_mode, prompt_family_version="v1", embedding_encoder_tag="qwen3"):
    version = str(prompt_family_version).lower()
    if version in {"v2", "v3"}:
        family_suffix = "v3" if version == "v3" else "v2"
        single_suffix = "_v3" if version == "v3" else ""
        names = {
            "expert_ego": f"glance_prompt_expert_ego{single_suffix}_{embedding_encoder_tag}_embed.pt",
            "expert_graph_following": f"glance_prompt_expert_graph_following{single_suffix}_{embedding_encoder_tag}_embed.pt",
            "expert_graph_follower": f"glance_prompt_expert_graph_follower{single_suffix}_{embedding_encoder_tag}_embed.pt",
            "expert_tweet": f"glance_prompt_expert_tweet{single_suffix}_{embedding_encoder_tag}_embed.pt",
            "expert_conflict": f"glance_prompt_expert_conflict{single_suffix}_{embedding_encoder_tag}_embed.pt",
            "expert_concat_v1": f"glance_prompt_expert_concat_{family_suffix}_{embedding_encoder_tag}_embed.pt",
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


def _residual_output_name(prompt_mode, residual_prompt_variant="base_as_hypothesis", embedding_encoder_tag="qwen3"):
    if prompt_mode != "residual_audit_v1":
        raise ValueError(f"Unsupported residual prompt mode: {prompt_mode}")
    variant = str(residual_prompt_variant or "base_as_hypothesis").strip().lower()
    if variant not in RESIDUAL_PROMPT_VARIANT_CHOICES:
        raise ValueError(f"Unsupported residual_prompt_variant: {variant}")
    return f"glance_residual_audit_v1_{variant}_{embedding_encoder_tag}_embed.pt"


def _dgp_output_name(prompt_mode, dgp_prompt_variant="target_fine_neighbor_coarse", embedding_encoder_tag="qwen3"):
    if prompt_mode not in DGP_PROMPT_MODES:
        raise ValueError(f"Unsupported DGP prompt mode: {prompt_mode}")
    variant = str(dgp_prompt_variant or "target_fine_neighbor_coarse").strip().lower()
    if prompt_mode == "dgp_predictor_v2" and variant in {"target_fine_neighbor_coarse", "target_only"}:
        variant = "norm_text_following_summary"
    if variant not in DGP_PROMPT_VARIANT_CHOICES:
        raise ValueError(f"Unsupported dgp_prompt_variant: {variant}")
    return f"{prompt_mode}_{variant}_{embedding_encoder_tag}_embed.pt"


def _default_output_path(
    dataset_path: Path,
    prompt_mode: str,
    prompt_family_version="v1",
    embedding_encoder_tag="qwen3",
    residual_prompt_variant="base_as_hypothesis",
    dgp_prompt_variant="target_fine_neighbor_coarse",
):
    if prompt_mode == "glance_concat_ego_hop1_hop2":
        return dataset_path / DEFAULT_OUTPUT_NAME
    if prompt_mode in EXPERT_PROMPT_MODES:
        return dataset_path / _expert_output_name(
            prompt_mode,
            prompt_family_version=prompt_family_version,
            embedding_encoder_tag=embedding_encoder_tag,
        )
    if prompt_mode in ULTRATAG_PROMPT_MODES:
        return dataset_path / f"ultratag_s_subgraph_v1_{embedding_encoder_tag}_embed.pt"
    if prompt_mode in RESIDUAL_AUDIT_PROMPT_MODES:
        return dataset_path / _residual_output_name(
            prompt_mode,
            residual_prompt_variant=residual_prompt_variant,
            embedding_encoder_tag=embedding_encoder_tag,
        )
    if prompt_mode in DGP_PROMPT_MODES:
        return dataset_path / _dgp_output_name(
            prompt_mode,
            dgp_prompt_variant=dgp_prompt_variant,
            embedding_encoder_tag=embedding_encoder_tag,
        )
    if prompt_mode in MHLGC_PROMPT_MODES:
        return dataset_path / f"{prompt_mode}_{embedding_encoder_tag}_embed.pt"
    return dataset_path / f"glance_qwen3_prompt_cache_{prompt_mode}.pt"


def _graph_prompt(
    direction_name,
    ego_record,
    neighbor_records,
    total_count,
    reciprocal_count,
):
    tweet_stats = _get_tweet_stats(ego_record, sample_size=5)
    return build_graph_prompt(
        direction_name,
        _target_account_text_for_llm(ego_record, tweet_stats, include_identity=True, brief=True),
        [_neighbor_card(record) for record in neighbor_records],
        total_count,
        reciprocal_count,
    )


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
    return build_graph_prompt_partitioned(
        direction_name,
        _profile_card(ego_record, brief=True),
        [_neighbor_card(record) for record in support_records],
        [_neighbor_card(record) for record in contrast_records],
        total_count,
        reciprocal_count,
        candidate_count,
        selected_count,
        mean_sim_support,
        mean_sim_contrast,
        reciprocal_ratio_selected,
    )


def _tweet_prompt(record, tweet_stats):
    return build_tweet_prompt(
        _profile_card(record, brief=True),
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
    )


def _ego_prompt(record, tweet_stats):
    prompt = build_ego_botsay_tweet_metadata_prompt(
        _profile_card(record, brief=False),
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
    )
    return "\n".join(
        [
            f"SYSTEM_MESSAGE:\n{prompt['system']}",
            f"USER_MESSAGE:\n{prompt['user']}",
            "ASSISTANT_RESPONSE:",
        ]
    )


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
    return (
        build_conflict_prompt(
            _profile_card(record, brief=False),
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
            [_neighbor_card(item) for item in following_records],
            [_neighbor_card(item) for item in follower_records],
            mismatch_hints,
        ),
        flags,
    )


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
    return (
        build_conflict_prompt_partitioned(
            _profile_card(record, brief=False),
            _tweet_behavior_summary(record, tweet_stats),
            _tweet_samples_block(tweet_stats),
            [_neighbor_card(item) for item in following_support_records],
            [_neighbor_card(item) for item in following_contrast_records],
            [_neighbor_card(item) for item in follower_support_records],
            [_neighbor_card(item) for item in follower_contrast_records],
            mismatch_hints,
        ),
        flags,
    )


def _ego_explain_generation_prompt(record, evidence_schema="summary", prompt_style="default"):
    structured = str(evidence_schema) == "structured_evidence_card"
    prompt = build_ego_explain_generation_prompt(
        _profile_card(record, brief=False),
        evidence_schema=evidence_schema,
        prompt_style=prompt_style,
    )
    return {
        "system": prompt["system"],
        "user": prompt["user"],
        "fallback_explanation": (
            _deterministic_ego_evidence_card_fallback(record)
            if structured
            else _deterministic_ego_explain_fallback(record)
        ),
        "prompt_role": prompt.get("prompt_role", "profile_explainer"),
    }


def _ego_embedding_prompt(explanation_text):
    return _expert_embedding_prompt("ego", explanation_text)


def _resolve_expert_prompt_bundle(args, records, edge_index, edge_type, target_node_ids=None, selection_features=None):
    num_nodes = len(records)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    mode = str(args.prompt_mode)
    prompt_family_version = str(getattr(args, "prompt_family_version", "v1")).lower()
    explanation_first = prompt_family_version in {"v2", "v3"}
    evidence_schema = "structured_evidence_card" if prompt_family_version == "v3" else "summary"
    explain_prompt_style = str(getattr(args, "explain_prompt_style", "default") or "default").strip().lower()
    policy = str(args.neighbor_sampling_policy).lower()
    if prompt_family_version not in PROMPT_FAMILY_VERSION_CHOICES:
        raise ValueError(f"Unsupported prompt_family_version: {prompt_family_version}")
    if explain_prompt_style not in EXPLAIN_PROMPT_STYLE_CHOICES:
        raise ValueError(
            f"Unsupported explain_prompt_style: {explain_prompt_style}. "
            f"Expected one of {EXPLAIN_PROMPT_STYLE_CHOICES}."
        )
    if explanation_first and policy == "center_induced_relation_aware":
        raise ValueError(
            f"prompt_expert_bundle_{prompt_family_version} keeps direction-split ranked neighbors as the mainline "
            "and does not expose center_induced_relation_aware."
        )
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
        selected_component_map_v2 if explanation_first else selected_component_map_v1
    )[mode]
    prompt_components = {name: [] for name in selected_components}
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
    if explanation_first:
        generation_components = set(selected_components)
        if "conflict" in selected_components:
            generation_components.update({"tweet", "graph_following", "graph_follower"})
        generation_rows_by_component = {name: [] for name in generation_components}
    elif False:
        generation_rows_by_component = {"ego": []}

    target_ids = list(target_node_ids) if target_node_ids is not None else list(range(num_nodes))
    for node_id in target_ids:
        record = records[node_id]
        if policy == "center_induced_relation_aware":
            if selection_features is None:
                raise ValueError("center_induced_relation_aware expert prompts require selection_features.")
            center_selection = select_center_induced_directional_neighbors(
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

        if explanation_first:
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
                    evidence_schema=evidence_schema,
                    prompt_style=explain_prompt_style,
                )
                generation_rows_by_component["graph_following"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "graph_following",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": generation_prompt["prompt_role"],
                    }
                )
                row["graph_following_generation_prompt"] = generation_prompt["user"]
            if "graph_follower" in generation_rows_by_component:
                generation_prompt = _graph_explain_generation_prompt(
                    "follower",
                    record,
                    follower_records,
                    follower_summary,
                    evidence_schema=evidence_schema,
                    prompt_style=explain_prompt_style,
                )
                generation_rows_by_component["graph_follower"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "graph_follower",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": generation_prompt["prompt_role"],
                    }
                )
                row["graph_follower_generation_prompt"] = generation_prompt["user"]
            if "tweet" in generation_rows_by_component:
                generation_prompt = _tweet_explain_generation_prompt(
                    record,
                    tweet_stats,
                    evidence_schema=evidence_schema,
                    prompt_style=explain_prompt_style,
                )
                generation_rows_by_component["tweet"].append(
                    {
                        "node_id": int(node_id),
                        "component_name": "tweet",
                        "system": generation_prompt["system"],
                        "user": generation_prompt["user"],
                        "fallback_explanation": generation_prompt["fallback_explanation"],
                        "prompt_role": generation_prompt["prompt_role"],
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

            if "ego" in selected_components:
                prompt_components["ego"].append(_ego_prompt(record, tweet_stats))
                row["ego_prompt"] = prompt_components["ego"][-1]

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
    elif prompt_family_version == "v3":
        prompt_family = "prompt_expert_bundle_v3"
        prompt_style = "expert_prompt_bundle_v3_structured_evidence_card"
        semantic_view_mode = "prompt_expert_bundle_v3"
        selection_policy_note = (
            "Direction-split fixed-quota ranking keeps follower and following separate; "
            "LLM generation extracts source-grounded structured evidence cards without final labels, probabilities, "
            "confidence scores, recommendations, or base-model correction instructions."
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
        "explain_prompt_style": explain_prompt_style,
        "prompt_family_version": prompt_family_version,
        "evidence_schema": evidence_schema,
        "evidence_card_fields": [
            "OBSERVED_BOT_LIKE_CUES",
            "OBSERVED_HUMAN_LIKE_CUES",
            "RELATION_AMBIGUITY",
            "POSSIBLE_BENIGN_EXPLANATION",
            "EVIDENCE_COVERAGE",
            "EVIDENCE_CONSISTENCY",
            "SUPPORTED_BY_SOURCE",
        ]
        if evidence_schema == "structured_evidence_card"
        else [],
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
                if explanation_first
                else list(EXPERT_COMPONENT_NAMES_WITH_STRUCTURED)
            ),
        }.items()},
        "selected_components": list(selected_components),
        "scalar_features": scalar_features,
        "prompt_rows": prompt_rows,
        "semantic_view_mode": semantic_view_mode,
        "selection_policy_note": selection_policy_note,
        "needs_conflict_generation": bool(explanation_first and "conflict" in selected_components),
    }


def _resolve_prompt_bundle(
    args,
    texts,
    edge_index,
    edge_type,
    classes,
    target_node_ids=None,
    selection_features=None,
    residual_base_outputs=None,
):
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
    elif mode in RESIDUAL_AUDIT_PROMPT_MODES:
        if policy not in {"auto", "directional_heuristic"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'directional_heuristic' or 'auto'.")
        records = [_parse_norm_user_record(raw_text, idx) for idx, raw_text in enumerate(texts)]
        bundle = _resolve_residual_audit_prompt_bundle(
            args,
            records,
            edge_index,
            edge_type,
            classes,
            target_node_ids=target_node_ids,
            base_outputs=residual_base_outputs,
        )
    elif mode in DGP_PROMPT_MODES:
        if policy not in {"auto", "directional_heuristic"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'directional_heuristic' or 'auto'.")
        if mode == "dgp_predictor_v2":
            bundle = _resolve_dgp_predictor_v2_prompt_bundle(
                args,
                texts,
                edge_index,
                edge_type,
                classes,
                target_node_ids=target_node_ids,
            )
        else:
            records = [_parse_norm_user_record(raw_text, idx) for idx, raw_text in enumerate(texts)]
            bundle = _resolve_dgp_predictor_prompt_bundle(
                args,
                records,
                edge_index,
                edge_type,
                classes,
                target_node_ids=target_node_ids,
            )
    elif mode in MHLGC_PROMPT_MODES:
        if policy not in {"center_induced_relation_aware"}:
            raise ValueError(
                f"{mode} requires --neighbor_sampling_policy center_induced_relation_aware "
                "so the hypergraph view is backed by --selection_embedding_path."
            )
        bundle = _resolve_mhlgc_prompt_bundle(
            args,
            texts,
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
    use_qwen_embedding_model = "qwen" in str(model_source).lower()
    model_dtype = (
        torch.float32
        if encoder_tag == "roberta_finetuned"
        else ("auto" if use_qwen_embedding_model else (torch.float16 if device.type == "cuda" else torch.float32))
    )
    model = AutoModel.from_pretrained(
        model_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
        torch_dtype=model_dtype,
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
                    text = _postprocess_generated_explanation(
                        text,
                        prompt_role=batch_rows[row_idx].get("prompt_role", ""),
                    )
                    if not text:
                        if bool(getattr(args, "explain_required", False)):
                            issue = "empty"
                        else:
                            fallback_count += 1
                            text = _fallback_explanation_text(batch_rows[row_idx])
                            issue = _explanation_quality_issue(text) if quality_gate else ""
                    else:
                        issue = _explanation_quality_issue(text) if quality_gate else ""
                    retry_count = 0
                    if issue and quality_gate:
                        max_retries = 2 if bool(getattr(args, "explain_required", False)) else 1
                        retry_user = str(batch_rows[row_idx].get("user", ""))
                        for retry_count in range(1, max_retries + 1):
                            retry_prompt = _format_chat_prompt(
                                tokenizer,
                                str(batch_rows[row_idx].get("system", "")),
                                _quality_retry_user_prompt(retry_user, issue, retry_count),
                            )
                            retry_raw = _generate_single_explanation_text(
                                model,
                                tokenizer,
                                retry_prompt,
                                args,
                                device,
                            )
                            retry_text = _postprocess_generated_explanation(
                                retry_raw,
                                prompt_role=batch_rows[row_idx].get("prompt_role", ""),
                            )
                            retry_issue = _explanation_quality_issue(retry_text)
                            quality_rejected[issue] += 1
                            if not retry_issue:
                                text = retry_text
                                issue = ""
                                break
                            issue = retry_issue
                    if issue:
                        quality_rejected[issue] += 1
                        fallback_count += 1
                        fallback_issue = issue
                        text = _fallback_explanation_text(batch_rows[row_idx])
                        generation_mode_row = "quality_fallback"
                    else:
                        fallback_issue = ""
                        generation_mode_row = "llm_generation"
                    explanations[batch_positions[row_idx]] = text
                    generated_count += 1
                    batch_sidecar_rows.append(
                        {
                            "node_id": int(batch_rows[row_idx]["node_id"]),
                            "component_name": component_name,
                            "prompt_role": batch_rows[row_idx].get("prompt_role", ""),
                            "prompt_hash": batch_hashes[row_idx],
                            "explanation": text,
                            "generation_mode": generation_mode_row,
                            "quality_retry_count": int(retry_count),
                            "quality_fallback_reason": fallback_issue,
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
    evidence_schema = str(prompt_bundle.get("evidence_schema", "summary") or "summary")
    row_by_node = {int(row["node_id"]): row for row in prompt_bundle.get("prompt_rows", [])}
    if component_name in prompt_bundle.get("selected_components", []):
        prompt_bundle["prompt_components"][component_name] = [
            _expert_embedding_prompt(component_name, item, evidence_schema=evidence_schema)
            for item in explanations
        ]
    for generation_row, explanation in zip(generation_rows, explanations):
        row = row_by_node.get(int(generation_row["node_id"]))
        if row is None:
            continue
        row[f"{component_name}_explanation"] = explanation
        row[f"{component_name}_embedding_prompt"] = _expert_embedding_prompt(
            component_name,
            explanation,
            evidence_schema=evidence_schema,
        )
    return prompt_bundle


def _prepare_explanation_first_conflict_generation_rows(prompt_bundle):
    evidence_schema = str(prompt_bundle.get("evidence_schema", "summary") or "summary")
    explain_prompt_style = str(prompt_bundle.get("explain_prompt_style", "default") or "default")
    generation_rows = []
    for row in prompt_bundle.get("prompt_rows", []):
        if "conflict" not in row.get("selected_components", []):
            continue
        generation_prompt = _conflict_explain_generation_prompt(
            row,
            evidence_schema=evidence_schema,
            prompt_style=explain_prompt_style,
        )
        row["conflict_generation_prompt"] = generation_prompt["user"]
        generation_rows.append(
            {
                "node_id": int(row["node_id"]),
                "component_name": "conflict",
                "system": generation_prompt["system"],
                "user": generation_prompt["user"],
                "fallback_explanation": generation_prompt["fallback_explanation"],
                "prompt_role": generation_prompt["prompt_role"],
            }
        )
    prompt_bundle.setdefault("generation_rows_by_component", {})["conflict"] = generation_rows
    return prompt_bundle


def _ultratag_profile_text(record):
    tweet_stats = _get_tweet_stats(record)
    pieces = [
        "PROFILE:",
        _profile_card(record, brief=True),
        "TWEET_BEHAVIOR:",
        _tweet_behavior_summary(record, tweet_stats),
        _tweet_samples_block(tweet_stats),
    ]
    return "\n".join(pieces)


def _ultratag_propagated_text(record, following_records, follower_records):
    lines = [
        "TARGET_NODE_TEXT:",
        _ultratag_profile_text(record),
        "PROPAGATED_NEIGHBOR_TEXT:",
        "FOLLOWING_NEIGHBORS:",
    ]
    if following_records:
        lines.extend(_neighbor_card(item) for item in following_records)
    else:
        lines.append("- None")
    lines.append("FOLLOWER_NEIGHBORS:")
    if follower_records:
        lines.extend(_neighbor_card(item) for item in follower_records)
    else:
        lines.append("- None")
    return "\n".join(lines)


def _ultratag_generation_prompt(task_name, propagated_text):
    task = str(task_name)
    system = "You prepare concise text augmentations for Twitter bot detection on text-attributed graphs."
    if task == "summary":
        user = (
            "Please summarize the target Twitter account and propagated neighbor context to improve "
            "suitability for bot-or-human node classification. Focus on identity, posting behavior, "
            "social context, and suspicious or benign cues. Output only the summary text in 3-5 sentences.\n\n"
            f"{propagated_text}"
        )
    elif task == "keywords":
        user = (
            "Please identify five short keywords or phrases from the target account and propagated neighbor "
            "context that are most relevant for bot-or-human classification. Output only a comma-separated list.\n\n"
            f"{propagated_text}"
        )
    elif task == "soft_label":
        user = (
            "Based on the target Twitter account and propagated neighbor context, predict the most appropriate "
            "classification label. Choose exactly one label from: human, bot. Output only the label.\n\n"
            f"{propagated_text}"
        )
    else:
        raise ValueError(f"Unsupported UltraTAG generation task: {task_name}")
    return {"system": system, "user": user}


def _ultratag_edge_prompt(row_a, row_b):
    system = "You estimate semantic social relatedness between two Twitter account nodes for graph refinement."
    user = (
        "You are provided with augmented text information for two Twitter accounts and their soft labels. "
        "Return the probability from 0 to 1 that an edge should exist between these two account nodes in a "
        "bot-detection graph because their evidence suggests meaningful social, topical, coordination, or "
        "audience relation. Output only one number between 0 and 1.\n\n"
        "NODE_A:\n"
        f"soft_label: {row_a.get('soft_label', 'unknown')}\n"
        f"{_truncate_chars(row_a.get('augmented_text', ''), 1800)}\n\n"
        "NODE_B:\n"
        f"soft_label: {row_b.get('soft_label', 'unknown')}\n"
        f"{_truncate_chars(row_b.get('augmented_text', ''), 1800)}"
    )
    return {"system": system, "user": user}


def _ultratag_parse_soft_label(text):
    lowered = str(text or "").strip().lower()
    if re.search(r"\bbot\b", lowered) and not re.search(r"\bhuman\b", lowered):
        return "bot"
    if re.search(r"\bhuman\b", lowered) and not re.search(r"\bbot\b", lowered):
        return "human"
    if lowered.startswith("bot"):
        return "bot"
    if lowered.startswith("human"):
        return "human"
    return "unknown"


def _ultratag_parse_probability(text):
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", str(text or ""))
    if not match:
        return 0.0
    value = float(match.group(0))
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return float(max(0.0, min(1.0, value)))


def _ultratag_clean_generation(task_name, text):
    text = str(text or "").strip()
    if task_name == "soft_label":
        return _ultratag_parse_soft_label(text)
    if task_name == "keywords":
        text = re.sub(r"(?is)^(keywords?|key phrases?)\s*:\s*", "", text).strip()
        parts = [part.strip(" \n\t-;:.") for part in re.split(r"[,;\n]", text) if part.strip(" \n\t-;:.")]
        return ", ".join(parts[:5]) if parts else "profile cues, posting behavior, social context, bot detection, account identity"
    if task_name == "edge_probability":
        return f"{_ultratag_parse_probability(text):.4f}"
    return text


def _ultratag_fallback(task_name, row):
    propagated = str(row.get("propagated_text", "") or "")
    if task_name == "summary":
        return _truncate_chars(_compact_whitespace(propagated), 520) or "The account has limited available text and social context."
    if task_name == "keywords":
        return "profile cues, posting behavior, social context, bot detection, account identity"
    if task_name == "soft_label":
        return "human"
    if task_name == "edge_probability":
        return "0.0000"
    return ""


def _generate_ultratag_text_rows(args, rows, task_name, device, generation_runtime, sidecar_path, wandb_run=None, step_base=0):
    task_name = str(task_name)
    if not rows:
        return [], {"total_count": 0, "generated_count": 0, "resumed_count": 0, "generation_mode": "skipped_empty"}
    tokenizer = generation_runtime.get("tokenizer") if generation_runtime else None
    model = generation_runtime.get("model") if generation_runtime else None
    prompt_texts = [
        _format_chat_prompt(tokenizer, row["system"], row["user"]) if tokenizer is not None else f"{row['system']}\n{row['user']}"
        for row in rows
    ]
    prompt_hashes = [_string_sha256(text) for text in prompt_texts]
    resumed = {}
    if sidecar_path and Path(sidecar_path).exists():
        for old in _read_jsonl(Path(sidecar_path)):
            if str(old.get("task_name", task_name)) != task_name:
                continue
            key_id = old.get("pair_key") if task_name == "edge_probability" else old.get("node_id")
            if key_id is None or "prompt_hash" not in old:
                continue
            resumed[(str(key_id), str(old["prompt_hash"]))] = str(old.get("output", "") or "")
    outputs = [None for _ in rows]
    pending = []
    for idx, (row, prompt_hash) in enumerate(zip(rows, prompt_hashes)):
        key_id = row.get("pair_key") if task_name == "edge_probability" else row.get("node_id")
        key = (str(key_id), str(prompt_hash))
        if key in resumed and str(resumed[key]).strip():
            outputs[idx] = _ultratag_clean_generation(task_name, resumed[key])
        else:
            pending.append((idx, row, prompt_texts[idx], prompt_hash))
    if pending and (tokenizer is None or model is None):
        if bool(getattr(args, "explain_required", False)):
            raise RuntimeError(f"UltraTAG-S task {task_name} requires --explain_model_path; no generation runtime is available.")
        for idx, row, _prompt, prompt_hash in pending:
            output = _ultratag_fallback(task_name, row)
            outputs[idx] = output
            if sidecar_path:
                payload = {
                    "task_name": task_name,
                    "node_id": int(row["node_id"]) if "node_id" in row else None,
                    "pair_key": row.get("pair_key", ""),
                    "prompt_hash": prompt_hash,
                    "output": output,
                    "generation_mode": "deterministic_fallback",
                }
                _append_jsonl(sidecar_path, [payload])
        return outputs, {
            "total_count": int(len(rows)),
            "generated_count": 0,
            "fallback_count": int(len(pending)),
            "resumed_count": int(len(rows) - len(pending)),
            "generation_mode": "deterministic_fallback",
            "sidecar_path": str(sidecar_path or ""),
        }
    batch_size = max(int(_effective_explain_batch_size(args, device)), 1)
    generated_count = 0
    start_time = time.time()
    log_every = max(int(getattr(args, "explain_log_every", 50) or 0), 0)
    for start in range(0, len(pending), batch_size):
        batch_items = pending[start : start + batch_size]
        batch_prompts = [item[2] for item in batch_items]
        batch = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=int(args.explain_max_input_length),
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=int(args.explain_max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        sidecar_rows = []
        for local_idx, generated_ids in enumerate(generated):
            idx, row, _prompt, prompt_hash = batch_items[local_idx]
            continuation = decoder_generation_continuation(generated_ids, batch["input_ids"][local_idx])
            raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            output = _ultratag_clean_generation(task_name, raw)
            if not output:
                output = _ultratag_fallback(task_name, row)
            outputs[idx] = output
            generated_count += 1
            sidecar_rows.append(
                {
                    "task_name": task_name,
                    "node_id": int(row["node_id"]) if "node_id" in row else None,
                    "pair_key": row.get("pair_key", ""),
                    "prompt_hash": prompt_hash,
                    "output": output,
                    "raw_output": raw,
                    "generation_mode": "llm_generation",
                }
            )
        if sidecar_path:
            _append_jsonl(sidecar_path, sidecar_rows)
        completed = int(len(rows) - len(pending) + generated_count)
        if log_every and (completed == len(rows) or generated_count % log_every == 0):
            elapsed = max(time.time() - start_time, 1e-6)
            progress = {
                "status": "ultratag_generation_progress",
                "task": task_name,
                "completed": completed,
                "total": int(len(rows)),
                "rows_per_second": float(generated_count) / elapsed,
            }
            print(progress)
            _wandb_log(
                wandb_run,
                {
                    f"precompute/ultratag/{task_name}/completed": completed,
                    f"precompute/ultratag/{task_name}/total": int(len(rows)),
                    f"precompute/ultratag/{task_name}/rows_per_second": progress["rows_per_second"],
                },
                step=int(step_base + completed),
            )
    return outputs, {
        "total_count": int(len(rows)),
        "generated_count": int(generated_count),
        "fallback_count": 0,
        "resumed_count": int(len(rows) - len(pending)),
        "generation_mode": "llm_generation" if generated_count else "llm_generation_resumed",
        "sidecar_path": str(sidecar_path or ""),
    }


def _load_base_embedding_tensor(path: Path, expected_rows: int):
    if not path:
        raise ValueError("UltraTAG-S subgraph precompute requires --ultratag_base_embedding_path.")
    loaded = torch.load(Path(path), map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("embeddings", "features", "x"):
            if key in loaded:
                loaded = loaded[key]
                break
    if not torch.is_tensor(loaded):
        loaded = torch.tensor(loaded)
    loaded = loaded.float().cpu()
    if loaded.dim() != 2:
        raise ValueError(f"Base embedding tensor must be [num_nodes, dim], got {tuple(loaded.shape)}.")
    if int(loaded.shape[0]) != int(expected_rows):
        raise ValueError(
            f"Base embedding rows ({int(loaded.shape[0])}) must match the UltraTAG output node count ({int(expected_rows)})."
        )
    return loaded


def _edge_set_from_tensors(edge_index, edge_type):
    edge_index = edge_index.detach().cpu().long()
    edge_type = edge_type.detach().cpu().long().view(-1)
    return {
        (int(edge_index[0, idx]), int(edge_index[1, idx]), int(edge_type[idx]))
        for idx in range(int(edge_index.shape[1]))
    }


def _simple_pagerank(nodes, undirected_adjacency, iterations=50, damping=0.85):
    nodes = [int(item) for item in nodes]
    if not nodes:
        return {}
    node_set = set(nodes)
    rank = {node: 1.0 / float(len(nodes)) for node in nodes}
    for _ in range(max(int(iterations), 1)):
        next_rank = {node: (1.0 - float(damping)) / float(len(nodes)) for node in nodes}
        dangling_mass = 0.0
        for src in nodes:
            neigh = [int(dst) for dst in undirected_adjacency.get(src, []) if int(dst) in node_set]
            if not neigh:
                dangling_mass += rank[src]
                continue
            share = float(damping) * rank[src] / float(len(neigh))
            for dst in neigh:
                next_rank[dst] += share
        if dangling_mass:
            share = float(damping) * dangling_mass / float(len(nodes))
            for node in nodes:
                next_rank[node] += share
        rank = next_rank
    return rank


def _run_ultratag_s_subgraph_precompute(
    args,
    *,
    dataset_path,
    output_texts,
    output_resolved_text_path,
    context_texts,
    context_resolved_text_path,
    labels,
    classes,
    output_edge_index,
    output_edge_type,
    context_edge_index,
    context_edge_type,
    output_variant_paths,
    context_variant_paths,
    graph_data_variant,
    context_graph_variant,
    target_node_bundle,
    output_path,
    embedding_encoder_tag,
    wandb_run,
):
    target_node_ids = [int(item) for item in target_node_bundle["target_node_ids"]]
    if not target_node_ids:
        raise ValueError("UltraTAG-S subgraph precompute requires at least one target routed node.")
    routed_nodes_path = str(target_node_bundle.get("routed_nodes_path", "") or "")
    target_split = str(getattr(args, "routed_nodes_split", "all") or "all").lower()
    target_scope = str(target_node_bundle.get("target_node_scope", ""))
    target_source = str(target_node_bundle.get("target_node_source", ""))
    if routed_nodes_path and target_split != "test":
        raise ValueError(
            "Routed-node UltraTAG-S adaptation is scoped to test routed_nodes; "
            "omit --routed_nodes_path and use --center_node_scope all_graph_nodes for full-graph augmentation."
        )

    output_node_count = int(len(output_texts))
    context_node_count = int(len(context_texts))
    labeled_node_count = int(labels.shape[0]) if torch.is_tensor(labels) and labels.dim() >= 1 else int(len(labels))
    invalid_context_targets = [int(item) for item in target_node_ids if int(item) >= int(context_node_count)]
    if invalid_context_targets:
        preview = invalid_context_targets[:10]
        raise ValueError(
            "UltraTAG-S target nodes must exist in the context graph/text universe. "
            f"context_node_count={context_node_count}, invalid targets={preview}"
            f"{' ...' if len(invalid_context_targets) > len(preview) else ''}."
        )
    context = _build_graph_context(context_edge_index, context_edge_type, context_node_count)
    record_cache = {}

    def _context_record(node_id):
        node_id = int(node_id)
        if node_id not in record_cache:
            record_cache[node_id] = _parse_norm_user_record(context_texts[node_id], node_id)
        return record_cache[node_id]

    target_set = set(target_node_ids)
    neighbor_quota = int(getattr(args, "ultratag_neighbor_quota", 3))
    rows = []
    prompt_rows = []
    for node_id in target_node_ids:
        following_ranked = _rank_directional_candidates(node_id, context["following"][node_id], context_texts, context)
        follower_ranked = _rank_directional_candidates(node_id, context["follower"][node_id], context_texts, context)
        following_ids = [int(item) for item in following_ranked[:neighbor_quota]]
        follower_ids = [int(item) for item in follower_ranked[:neighbor_quota]]
        propagated_text = _ultratag_propagated_text(
            _context_record(node_id),
            [_context_record(idx) for idx in following_ids],
            [_context_record(idx) for idx in follower_ids],
        )
        row = {
            "node_id": int(node_id),
            "following_ids": following_ids,
            "follower_ids": follower_ids,
            "propagated_text": propagated_text,
            "count_following": int(len(set(context["following"][node_id]))),
            "count_follower": int(len(set(context["follower"][node_id]))),
        }
        rows.append(row)
        prompt_rows.append(
            {
                "node_id": int(node_id),
                "following_ids": following_ids,
                "follower_ids": follower_ids,
                "propagated_text": propagated_text,
            }
        )

    device = _device_from_args(str(args.device))
    generation_runtime = _resolve_generation_runtime(args, device)
    if generation_runtime is None and bool(getattr(args, "explain_required", False)):
        raise ValueError("UltraTAG-S generation requires --explain_model_path when --explain_required is set.")
    sidecar_dir = Path(getattr(args, "explain_component_cache_dir", None) or output_path.parent)
    ensure_dir(sidecar_dir)
    generation_summaries = {}
    for task_idx, task_name in enumerate(("summary", "keywords", "soft_label")):
        generation_rows = []
        for row in rows:
            prompt = _ultratag_generation_prompt(task_name, row["propagated_text"])
            generation_rows.append(
                {
                    "node_id": int(row["node_id"]),
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "propagated_text": row["propagated_text"],
                }
            )
        sidecar_path = sidecar_dir / f"{output_path.stem}_{task_name}.jsonl"
        outputs, summary = _generate_ultratag_text_rows(
            args,
            generation_rows,
            task_name,
            device,
            generation_runtime,
            sidecar_path,
            wandb_run=wandb_run,
            step_base=1000 * (task_idx + 1),
        )
        generation_summaries[task_name] = summary
        for row, value in zip(rows, outputs):
            row[task_name] = value

    if generation_runtime is not None:
        generation_runtime["model"] = None
        generation_runtime["tokenizer"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for row in rows:
        row["augmented_text"] = "\n".join(
            [
                "ULTRATAG_TEXT_PROPAGATION:",
                row["propagated_text"],
                "ULTRATAG_SUMMARY:",
                row.get("summary", ""),
                "ULTRATAG_KEYWORDS:",
                row.get("keywords", ""),
                "ULTRATAG_SOFT_LABEL:",
                row.get("soft_label", "unknown"),
            ]
        )

    tokenizer, model, resolved_embedding_model_source, embedding_pooling_mode, simteg_checkpoint_summary = _load_embedding_model(args, device)
    encoded_target = _encode_texts(
        model,
        tokenizer,
        [row["augmented_text"] for row in rows],
        device,
        int(args.batch_size),
        int(args.max_length_hop),
        bool(args.normalize),
        embedding_pooling_mode,
    ).float().cpu()
    del model
    base_embeddings = _load_base_embedding_tensor(getattr(args, "ultratag_base_embedding_path", None), output_node_count)
    if int(base_embeddings.shape[1]) != int(encoded_target.shape[1]):
        raise ValueError(
            "UltraTAG-S encoded dimension must match --ultratag_base_embedding_path dimension for row replacement: "
            f"{int(encoded_target.shape[1])} vs {int(base_embeddings.shape[1])}."
        )
    augmented_embeddings = base_embeddings.clone()
    target_index_tensor = torch.tensor(target_node_ids, dtype=torch.long)
    augmented_embeddings[target_index_tensor] = encoded_target
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    augmented_embeddings = augmented_embeddings.to(save_dtype).contiguous()

    virtual_edge_policy = str(getattr(args, "ultratag_virtual_edge_policy", "same_soft_label_cosine")).lower()
    tau1 = float(getattr(args, "ultratag_tau1", 0.8))
    tau2 = float(getattr(args, "ultratag_tau2", 0.5))
    relation_type = int(getattr(args, "ultratag_virtual_edge_relation", 1))
    existing_edges = _edge_set_from_tensors(output_edge_index, output_edge_type)
    new_edges = set()
    virtual_edge_rows = []
    if virtual_edge_policy == "same_soft_label_cosine":
        normed = F.normalize(encoded_target.float(), p=2, dim=1, eps=1e-12)
        sim = normed @ normed.T
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if str(rows[i].get("soft_label", "unknown")) != str(rows[j].get("soft_label", "unknown")):
                    continue
                if str(rows[i].get("soft_label", "unknown")) == "unknown":
                    continue
                score = float(sim[i, j].item())
                if score <= tau1:
                    continue
                a = int(rows[i]["node_id"])
                b = int(rows[j]["node_id"])
                for src, dst in ((a, b), (b, a)):
                    edge = (src, dst, relation_type)
                    if edge not in existing_edges:
                        new_edges.add(edge)
                virtual_edge_rows.append(
                    {
                        "src": a,
                        "dst": b,
                        "soft_label": str(rows[i].get("soft_label", "unknown")),
                        "cosine_similarity": score,
                        "accepted": True,
                        "policy": virtual_edge_policy,
                    }
                )
        del sim
    elif virtual_edge_policy != "none":
        raise ValueError(f"Unsupported --ultratag_virtual_edge_policy: {virtual_edge_policy}")

    augmented_undirected = {node: set(context["undirected"][node]).intersection(target_set) for node in target_node_ids}
    for src, dst, _rel in new_edges:
        if src in target_set and dst in target_set:
            augmented_undirected.setdefault(src, set()).add(dst)
            augmented_undirected.setdefault(dst, set()).add(src)
    pagerank = _simple_pagerank(target_node_ids, augmented_undirected, iterations=50, damping=0.85)
    select_count = max(1, int(math.ceil(float(getattr(args, "ultratag_pagerank_ratio", 0.10)) * float(len(target_node_ids)))))
    selected_nodes = [
        int(node)
        for node, _score in sorted(pagerank.items(), key=lambda item: (-float(item[1]), int(item[0])))[:select_count]
    ]
    row_by_node = {int(row["node_id"]): row for row in rows}
    selected_pairs = []
    max_pairs = int(getattr(args, "ultratag_edge_reconfig_max_pairs", 512))
    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            if len(selected_pairs) >= max_pairs:
                break
            selected_pairs.append((selected_nodes[i], selected_nodes[j]))
        if len(selected_pairs) >= max_pairs:
            break
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edge_reconfig_rows = []
    edge_reconfig_summary = {"total_count": 0, "generated_count": 0, "resumed_count": 0, "generation_mode": "disabled"}
    if bool(getattr(args, "ultratag_edge_reconfig", True)) and selected_pairs:
        if generation_runtime is not None and (generation_runtime.get("model") is None or generation_runtime.get("tokenizer") is None):
            generation_runtime = _resolve_generation_runtime(args, device)
        generation_rows = []
        for src, dst in selected_pairs:
            prompt = _ultratag_edge_prompt(row_by_node[src], row_by_node[dst])
            generation_rows.append(
                {
                    "pair_key": f"{int(src)}-{int(dst)}",
                    "src": int(src),
                    "dst": int(dst),
                    "system": prompt["system"],
                    "user": prompt["user"],
                }
            )
        outputs, edge_reconfig_summary = _generate_ultratag_text_rows(
            args,
            generation_rows,
            "edge_probability",
            device,
            generation_runtime,
            sidecar_dir / f"{output_path.stem}_edge_reconfig.jsonl",
            wandb_run=wandb_run,
            step_base=5000,
        )
        for gen_row, value in zip(generation_rows, outputs):
            prob = _ultratag_parse_probability(value)
            accepted = bool(prob > tau2)
            src = int(gen_row["src"])
            dst = int(gen_row["dst"])
            if accepted:
                for edge_src, edge_dst in ((src, dst), (dst, src)):
                    edge = (edge_src, edge_dst, relation_type)
                    if edge not in existing_edges:
                        new_edges.add(edge)
            edge_reconfig_rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "probability": prob,
                    "accepted": accepted,
                }
            )

    edge_index_cpu = output_edge_index.detach().cpu().long()
    edge_type_cpu = output_edge_type.detach().cpu().long().view(-1)
    sorted_new_edges = sorted(new_edges)
    if sorted_new_edges:
        add_index = torch.tensor([[src for src, _dst, _rel in sorted_new_edges], [dst for _src, dst, _rel in sorted_new_edges]], dtype=torch.long)
        add_type = torch.tensor([rel for _src, _dst, rel in sorted_new_edges], dtype=torch.long)
        augmented_edge_index = torch.cat([edge_index_cpu, add_index], dim=1).contiguous()
        augmented_edge_type = torch.cat([edge_type_cpu, add_type], dim=0).contiguous()
    else:
        augmented_edge_index = edge_index_cpu.contiguous()
        augmented_edge_type = edge_type_cpu.contiguous()
    edge_index_path = output_path.with_name(f"{output_path.stem}_edge_index.pt")
    edge_type_path = output_path.with_name(f"{output_path.stem}_edge_type.pt")
    write_torch(edge_index_path, augmented_edge_index)
    write_torch(edge_type_path, augmented_edge_type)

    payload = {
        "embeddings": augmented_embeddings,
        "semantic_view_mode": "ultratag_s_subgraph_v1",
        "prompt_family": "ultratag_s_subgraph_adaptation",
        "target_node_ids": target_index_tensor,
        "target_node_mask": torch.zeros((output_node_count,), dtype=torch.bool),
        "target_node_scope": target_scope,
        "target_node_source": target_source,
        "routed_nodes_path": routed_nodes_path,
        "routed_nodes_split": str(target_node_bundle.get("routed_nodes_split", "")),
        "full_graph_node_count": int(output_node_count),
        "output_node_count": int(output_node_count),
        "context_graph_node_count": int(context_node_count),
        "labeled_node_count": int(labeled_node_count),
        "embedding_encoder_tag": _infer_embedding_encoder_tag(resolved_embedding_model_source),
        "base_embedding_path": str(getattr(args, "ultratag_base_embedding_path", "")),
    }
    payload["target_node_mask"][target_index_tensor] = True
    write_torch(output_path, payload)
    prompt_sidecar_path = output_path.with_name(f"{output_path.stem}_prompts.jsonl")
    augmentation_sidecar_path = output_path.with_name(f"{output_path.stem}_augmentations.jsonl")
    virtual_edge_sidecar_path = output_path.with_name(f"{output_path.stem}_virtual_edges.jsonl")
    edge_reconfig_decision_sidecar_path = output_path.with_name(f"{output_path.stem}_edge_reconfig_decisions.jsonl")
    _write_jsonl(prompt_sidecar_path, prompt_rows)
    _write_jsonl(augmentation_sidecar_path, rows)
    _write_jsonl(virtual_edge_sidecar_path, virtual_edge_rows)
    _write_jsonl(edge_reconfig_decision_sidecar_path, edge_reconfig_rows)

    tensor_hashes = {
        "embeddings": tensor_sha256(payload["embeddings"]),
        "target_node_ids": tensor_sha256(payload["target_node_ids"]),
        "target_node_mask": tensor_sha256(payload["target_node_mask"]),
        "edge_index": tensor_sha256(augmented_edge_index),
        "edge_type": tensor_sha256(augmented_edge_type),
    }
    manifest = {
        "status": "completed",
        "dataset": str(args.dataset),
        "dataset_path": str(dataset_path),
        "method_family": "ultratag_s_subgraph_adaptation",
        "prompt_mode": str(args.prompt_mode),
        "semantic_view_mode": "ultratag_s_subgraph_v1",
        "paper_reference": "UltraTAG-S arXiv:2504.02343",
        "paper_faithful_scope": (
            "UltraTAG-S data/text augmentation adaptation with configurable target scope and structure policy; "
            "not the full UltraTAG-S dual-GNN structure-learning reproduction"
        ),
        "ultratag_modules": [
            "text_propagation",
            "text_augmentation_summary_keywords_soft_label",
            "pagerank_node_selector",
            "lm_based_representation_row_replacement",
        ]
        + (["virtual_edge_generator"] if virtual_edge_policy != "none" else [])
        + (["llm_edge_reconfigurator"] if bool(getattr(args, "ultratag_edge_reconfig", True)) else []),
        "target_node_count": int(len(target_node_ids)),
        "target_node_scope": target_scope,
        "target_node_source": target_source,
        "target_node_split": str(target_node_bundle.get("routed_nodes_split", "")),
        "target_node_ids_preview": [int(item) for item in target_node_ids[:32]],
        "routed_nodes_path": routed_nodes_path,
        "routed_nodes_split": str(target_node_bundle.get("routed_nodes_split", "")),
        "text_path": str(output_resolved_text_path),
        "context_text_path": str(context_resolved_text_path),
        "edge_index_path": str(output_variant_paths["edge_index_path"]),
        "edge_type_path": str(output_variant_paths["edge_type_path"]),
        "context_edge_index_path": str(context_variant_paths["edge_index_path"]),
        "context_edge_type_path": str(context_variant_paths["edge_type_path"]),
        "output_path": str(output_path),
        "base_embedding_path": str(getattr(args, "ultratag_base_embedding_path", "")),
        "embedding_model_path": str(resolved_embedding_model_source),
        "embedding_encoder_tag": _infer_embedding_encoder_tag(resolved_embedding_model_source),
        "embedding_pooling_mode": embedding_pooling_mode,
        "finetuned_roberta_checkpoint_path": str((simteg_checkpoint_summary or {}).get("checkpoint_path", "")),
        "finetuned_roberta_checkpoint_load": dict(simteg_checkpoint_summary or {}),
        "explain_model_path": str(getattr(args, "explain_model_path", "") or ""),
        "neighbor_quota": int(neighbor_quota),
        "virtual_edge_policy": virtual_edge_policy,
        "tau1": float(tau1),
        "tau2": float(tau2),
        "pagerank_ratio": float(getattr(args, "ultratag_pagerank_ratio", 0.10)),
        "pagerank_selected_count": int(len(selected_nodes)),
        "pagerank_selected_nodes": selected_nodes,
        "edge_reconfig_enabled": bool(getattr(args, "ultratag_edge_reconfig", True)),
        "edge_reconfig_max_pairs": int(max_pairs),
        "edge_reconfig_pair_count": int(len(selected_pairs)),
        "virtual_edge_relation_type": int(relation_type),
        "virtual_edge_relation_policy": (
            "disabled; no soft-label-derived virtual edges are added"
            if virtual_edge_policy == "none"
            else (
                "adds symmetric edges inside the target-node induced subgraph using an existing TwiBot relation id; "
                "default relation_1 avoids adding a third unseen RGCN relation"
            )
        ),
        "original_edge_count": int(edge_index_cpu.shape[1]),
        "context_edge_count": int(context_edge_index.detach().cpu().shape[1]),
        "added_edge_count": int(len(sorted_new_edges)),
        "augmented_edge_count": int(augmented_edge_index.shape[1]),
        "augmented_edge_index_path": str(edge_index_path),
        "augmented_edge_type_path": str(edge_type_path),
        "prompt_sidecar_path": str(prompt_sidecar_path),
        "augmentation_sidecar_path": str(augmentation_sidecar_path),
        "virtual_edge_sidecar_path": str(virtual_edge_sidecar_path),
        "edge_reconfig_decision_sidecar_path": str(edge_reconfig_decision_sidecar_path),
        "generation_sidecars": {
            "summary": str(sidecar_dir / f"{output_path.stem}_summary.jsonl"),
            "keywords": str(sidecar_dir / f"{output_path.stem}_keywords.jsonl"),
            "soft_label": str(sidecar_dir / f"{output_path.stem}_soft_label.jsonl"),
            "edge_reconfig": str(sidecar_dir / f"{output_path.stem}_edge_reconfig.jsonl"),
        },
        "generation_summaries": {
            **generation_summaries,
            "edge_reconfig": edge_reconfig_summary,
        },
        "payload_row_layout": (
            (
                "full_graph_augmented_embeddings"
                if str(graph_data_variant) == "full_graph_support"
                else "labeled_augmented_embeddings"
            )
            if len(target_node_ids) == int(output_node_count)
            else (
                "full_graph_base_embedding_with_target_rows_replaced"
                if str(graph_data_variant) == "full_graph_support"
                else "labeled_base_embedding_with_target_rows_replaced"
            )
        ),
        "graph_data_variant": graph_data_variant,
        "output_graph_variant": graph_data_variant,
        "context_graph_variant": context_graph_variant,
        "classes": classes,
        "save_dtype": str(args.save_dtype),
        "tensor_sha256": tensor_hashes,
    }
    write_json(output_path.with_name(f"{output_path.stem}_manifest.json"), manifest)
    _wandb_log(
        wandb_run,
        {
            "precompute/stage_code": 2,
            "precompute/completed": 1,
            "precompute/ultratag/target_node_count": int(len(target_node_ids)),
            "precompute/ultratag/added_edge_count": int(len(sorted_new_edges)),
        },
        step=999,
    )
    _finish_precompute_wandb(wandb_run, exit_code=0)
    print(
        {
            "status": "completed",
            "output_path": str(output_path),
            "prompt_mode": str(args.prompt_mode),
            "target_node_count": int(len(target_node_ids)),
            "added_edge_count": int(len(sorted_new_edges)),
            "augmented_edge_index_path": str(edge_index_path),
            "augmented_edge_type_path": str(edge_type_path),
        }
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute GLANCE-style, relation-aware, expert, DGP, or MH-LGC prompt caches.")
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
            "When set, supported precompute modes run only on those graph-global node ids."
        ),
    )
    parser.add_argument(
        "--routed_nodes_split",
        choices=("all", "train", "valid", "val", "test"),
        default="all",
        help=(
            "Optional split selector for mapping-style routed node files. "
            "Routed-node UltraTAG-S adaptation is restricted to --routed_nodes_split test; "
            "omit --routed_nodes_path and use --center_node_scope all_graph_nodes for full-graph UltraTAG text augmentation."
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
            "expert_* + --prompt_family_version v2/v3 defaults to the SimTeG finetuned RoBERTa alias "
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
            "If omitted, v2/v3 expert modes search for TwiBot-20_seed_<seed>/checkpoints/LM_pretrain/best.pkl."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_mode", choices=PROMPT_MODE_CHOICES, default="glance_concat_ego_hop1_hop2")
    parser.add_argument("--prompt_family_version", choices=PROMPT_FAMILY_VERSION_CHOICES, default="v1")
    parser.add_argument(
        "--residual_base_outputs_path",
        type=Path,
        default=None,
        help=(
            "Frozen SimTeG graph-detector outputs.pt used by residual_audit_v1 prompts. "
            "Required for base_as_hypothesis/base_as_assertion variants; labels from this file are not inserted into prompts."
        ),
    )
    parser.add_argument(
        "--residual_prompt_variant",
        choices=RESIDUAL_PROMPT_VARIANT_CHOICES,
        default="base_as_hypothesis",
        help=(
            "Residual-audit prompt framing: expose the base prediction as a fallible hypothesis, "
            "hide it for no_base ablation, or expose it as a strong prior for anchoring diagnostics."
        ),
    )
    parser.add_argument(
        "--residual_include_neighbor_base_distribution",
        type=_parse_bool,
        default=True,
        help=(
            "For residual_audit_v1 with an exposed base prediction, include compact base-predicted class "
            "distributions over following/follower/1-hop neighbors. True labels are never used."
        ),
    )
    parser.add_argument(
        "--dgp_prompt_variant",
        choices=DGP_PROMPT_VARIANT_CHOICES,
        default="target_fine_neighbor_coarse",
        help=(
            "DGP-style predictor prompt layout. v1 supports target_fine_neighbor_coarse and target_only; "
            "v2 supports norm_text_following_summary, norm_text_follower_summary, "
            "norm_text_following_follower_summary, and norm_text_target_only."
        ),
    )
    parser.add_argument(
        "--dgp_neighbor_summary_k",
        type=int,
        default=5,
        help=(
            "Top-K directional neighbors to summarize for dgp_predictor_v2. "
            "Mainline uses K=5 following neighbors; follower variants use the same K for ablation."
        ),
    )
    parser.add_argument(
        "--explain_prompt_style",
        choices=EXPLAIN_PROMPT_STYLE_CHOICES,
        default="default",
        help=(
            "Prompt style for explanation-first expert generation in v2/v3. "
            "'botsay' reuses BotSay-style label-first / explanation-after framing without importing labeled exemplars or neighbor labels."
        ),
    )
    parser.add_argument("--neighbor_sampling_policy", choices=NEIGHBOR_SAMPLING_CHOICES, default="auto")
    parser.add_argument(
        "--selection_embedding_path",
        type=Path,
        default=None,
        help=(
            "Node-aligned semantic feature tensor used by center_induced_relation_aware prompt selection "
            "and by mhlgc_llm_guide to construct the HyperScan-style KNN hypergraph view."
        ),
    )
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
    parser.add_argument(
        "--ultratag_base_embedding_path",
        type=Path,
        default=None,
        help=(
            "Base full-graph embedding tensor used by ultratag_s_subgraph_v1. "
            "Rows selected by --center_node_scope or --routed_nodes_path are replaced by UltraTAG-S augmented text embeddings."
        ),
    )
    parser.add_argument("--ultratag_neighbor_quota", type=int, default=3)
    parser.add_argument("--ultratag_tau1", type=float, default=0.8)
    parser.add_argument("--ultratag_tau2", type=float, default=0.5)
    parser.add_argument("--ultratag_pagerank_ratio", type=float, default=0.10)
    parser.add_argument("--ultratag_edge_reconfig_max_pairs", type=int, default=512)
    parser.add_argument("--ultratag_edge_reconfig", type=_parse_bool, default=True)
    parser.add_argument(
        "--ultratag_virtual_edge_policy",
        choices=("same_soft_label_cosine", "none"),
        default="same_soft_label_cosine",
        help=(
            "Virtual-edge construction policy for ultratag_s_subgraph_v1. "
            "Use 'none' to disable soft-label-derived cosine edges and keep the graph structure unchanged "
            "unless --ultratag_edge_reconfig is explicitly enabled."
        ),
    )
    parser.add_argument("--ultratag_virtual_edge_relation", type=int, default=1)
    return parser


def run(args):
    _apply_prompt_expert_explanation_first_encoder_defaults(args)
    wandb_run = _setup_precompute_wandb(args)
    if (
        getattr(args, "routed_nodes_path", None)
        and not str(getattr(args, "prompt_mode", "")).startswith("expert_")
        and str(getattr(args, "prompt_mode", "")) not in ULTRATAG_PROMPT_MODES
        and str(getattr(args, "prompt_mode", "")) not in RESIDUAL_AUDIT_PROMPT_MODES
        and str(getattr(args, "prompt_mode", "")) not in DGP_PROMPT_MODES
        and str(getattr(args, "prompt_mode", "")) not in MHLGC_PROMPT_MODES
    ):
        raise ValueError(
            "--routed_nodes_path is only supported for expert_* prompt modes, ultratag_s_subgraph_v1, "
            "residual_audit_v1, dgp_predictor_v1, and mhlgc_llm_guide."
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
    labels_path = dataset_path / "labels.pt"
    labels = torch.load(labels_path, map_location="cpu")
    labeled_node_count = int(labels.shape[0]) if torch.is_tensor(labels) and labels.dim() >= 1 else int(len(labels))

    is_ultratag_mode = str(getattr(args, "prompt_mode", "")) in ULTRATAG_PROMPT_MODES
    if is_ultratag_mode:
        output_variant_value = getattr(args, "graph_data_variant", "labeled")
        context_variant_value = getattr(args, "context_graph_variant", None) or output_variant_value
        output_variant_paths = _resolve_graph_variant_paths(dataset_path, output_variant_value, args.text_path)
        output_texts, output_resolved_text_path = _load_texts(dataset_path, output_variant_paths["text_path"])
        if not output_texts:
            raise ValueError("No node texts found for UltraTAG-S output graph precompute.")
        context_text_override = args.text_path if str(context_variant_value).lower() == str(output_variant_value).lower() else None
        context_variant_paths = _resolve_graph_variant_paths(dataset_path, context_variant_value, context_text_override)
        context_texts, context_resolved_text_path = _load_texts(dataset_path, context_variant_paths["text_path"])
        if not context_texts:
            raise ValueError("No node texts found for UltraTAG-S context graph precompute.")
        graph_data_variant = output_variant_paths["variant"]
        context_graph_variant = context_variant_paths["variant"]
        full_node_count = int(len(output_texts))
        target_node_bundle = _resolve_target_node_ids(
            args,
            labeled_node_count=labeled_node_count,
            full_node_count=full_node_count,
        )
    else:
        graph_variant_value = getattr(args, "context_graph_variant", None) or getattr(args, "graph_data_variant", "labeled")
        variant_paths = _resolve_graph_variant_paths(dataset_path, graph_variant_value, args.text_path)
        texts, resolved_text_path = _load_texts(dataset_path, variant_paths["text_path"])
        if not texts:
            raise ValueError("No node texts found for Qwen prompt precompute.")
        graph_data_variant = variant_paths["variant"]
        full_node_count = int(len(texts))
        target_node_bundle = _resolve_target_node_ids(
            args,
            labeled_node_count=labeled_node_count,
            full_node_count=full_node_count,
        )
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

    if is_ultratag_mode:
        output_edge_index = torch.load(output_variant_paths["edge_index_path"], map_location="cpu")
        output_edge_type = torch.load(output_variant_paths["edge_type_path"], map_location="cpu")
        context_edge_index = torch.load(context_variant_paths["edge_index_path"], map_location="cpu")
        context_edge_type = torch.load(context_variant_paths["edge_type_path"], map_location="cpu")
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
                residual_prompt_variant=str(getattr(args, "residual_prompt_variant", "base_as_hypothesis")),
                dgp_prompt_variant=str(getattr(args, "dgp_prompt_variant", "target_fine_neighbor_coarse")),
            )
        )
        if output_path.exists() and not bool(args.overwrite):
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or a different --output_path.")
        ensure_dir(output_path.parent)
        return _run_ultratag_s_subgraph_precompute(
            args,
            dataset_path=dataset_path,
            output_texts=output_texts,
            output_resolved_text_path=output_resolved_text_path,
            context_texts=context_texts,
            context_resolved_text_path=context_resolved_text_path,
            labels=labels,
            classes=classes,
            output_edge_index=output_edge_index,
            output_edge_type=output_edge_type,
            context_edge_index=context_edge_index,
            context_edge_type=context_edge_type,
            output_variant_paths=output_variant_paths,
            context_variant_paths=context_variant_paths,
            graph_data_variant=graph_data_variant,
            context_graph_variant=context_graph_variant,
            target_node_bundle=target_node_bundle,
            output_path=output_path,
            embedding_encoder_tag=embedding_encoder_tag,
            wandb_run=wandb_run,
        )
    edge_index = torch.load(variant_paths["edge_index_path"], map_location="cpu")
    edge_type = torch.load(variant_paths["edge_type_path"], map_location="cpu")
    selection_feature_bundle = None
    if str(args.neighbor_sampling_policy).lower() == "center_induced_relation_aware":
        selection_feature_bundle = resolve_selection_feature_bundle(
            args,
            dataset_path,
            graph_variant=graph_data_variant,
            graph_node_count=full_node_count,
            labeled_node_count=labeled_node_count,
        )
    residual_base_outputs = None
    if str(getattr(args, "prompt_mode", "")) in RESIDUAL_AUDIT_PROMPT_MODES:
        residual_variant = str(getattr(args, "residual_prompt_variant", "base_as_hypothesis") or "base_as_hypothesis").strip().lower()
        if residual_variant != "no_base":
            if getattr(args, "residual_base_outputs_path", None) is None:
                raise ValueError(
                    f"--residual_base_outputs_path is required for --prompt_mode {args.prompt_mode} "
                    f"with --residual_prompt_variant {residual_variant}."
                )
            residual_base_outputs = _load_residual_base_outputs(
                Path(args.residual_base_outputs_path),
                target_node_ids=target_node_ids,
            )
    prompt_bundle = _resolve_prompt_bundle(
        args,
        texts,
        edge_index,
        edge_type,
        classes,
        target_node_ids=target_node_ids,
        selection_features=selection_feature_bundle["features"] if selection_feature_bundle is not None else None,
        residual_base_outputs=residual_base_outputs,
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
            residual_prompt_variant=str(getattr(args, "residual_prompt_variant", "base_as_hypothesis")),
            dgp_prompt_variant=str(getattr(args, "dgp_prompt_variant", "target_fine_neighbor_coarse")),
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
            prompt_bundle = _prepare_explanation_first_conflict_generation_rows(prompt_bundle)
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

    if str(args.prompt_mode) == "dgp_predictor_v2" and bool(prompt_bundle.get("needs_dgp_v2_context_generation")):
        generation_runtime = _resolve_generation_runtime(args, device)
        dgp_v2_generation_order = [
            ("dgp_neighbor_summary", _attach_dgp_v2_neighbor_summaries),
            ("dgp_context_summary", _attach_dgp_v2_context_summaries),
        ]
        for component_name, attach_fn in dgp_v2_generation_order:
            generation_rows = list(prompt_bundle.get("generation_rows_by_component", {}).get(component_name, []))
            if not generation_rows:
                if component_name == "dgp_neighbor_summary":
                    prompt_bundle = attach_fn(prompt_bundle, [])
                continue
            component_sidecar_path = explanation_cache_dir / f"{output_path.stem}_{component_name}_explanations.jsonl"
            component_explain_sidecar_paths[component_name] = str(component_sidecar_path)
            explanations, component_summary = _generate_component_explanations(
                args,
                generation_rows,
                device,
                generation_runtime=generation_runtime,
                component_name=component_name,
                sidecar_path=component_sidecar_path,
                wandb_run=wandb_run,
                wandb_step_base=7000 + 1000 * len(explain_summary_by_component),
            )
            explain_summary_by_component[component_name] = component_summary
            prompt_bundle = attach_fn(prompt_bundle, explanations)
            _wandb_log(
                wandb_run,
                {
                    "precompute/dgp_v2_component_done": 1,
                    f"precompute/dgp_v2/{component_name}/rows": int(len(generation_rows)),
                    f"precompute/dgp_v2/{component_name}/generated_count": int(component_summary.get("generated_count", 0)),
                    f"precompute/dgp_v2/{component_name}/fallback_count": int(component_summary.get("fallback_count", 0)),
                    f"precompute/dgp_v2/{component_name}/generation_mode_code": 1
                    if str(component_summary.get("generation_mode", "")) == "llm_generation"
                    else 0,
                },
                step=30 + len(explain_summary_by_component),
            )
            explain_rows_for_sidecar.extend(
                {
                    "node_id": int(row["node_id"]),
                    "component_name": component_name,
                    "prompt_role": row.get("prompt_role", ""),
                    "summary_direction": row.get("summary_direction", ""),
                    "neighbor_node_id": int(row.get("neighbor_node_id", -1)) if "neighbor_node_id" in row else None,
                    "neighbor_rank": int(row.get("neighbor_rank", 0)) if "neighbor_rank" in row else None,
                    "explanation": explanation,
                }
                for row, explanation in zip(generation_rows, explanations)
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
        "prompt_family_version": str(prompt_bundle.get("prompt_family_version", prompt_family_version)),
        "residual_prompt_variant": str(prompt_bundle.get("residual_prompt_variant", "")),
        "dgp_prompt_variant": str(prompt_bundle.get("dgp_prompt_variant", "")),
        "residual_base_outputs_path": str(prompt_bundle.get("base_outputs_path", "")),
        "residual_include_neighbor_base_distribution": bool(
            prompt_bundle.get("include_neighbor_base_distribution", False)
        ),
        "evidence_schema": str(prompt_bundle.get("evidence_schema", "summary")),
        "norm_user_text_rendering": str(prompt_bundle.get("norm_user_text_rendering", "")),
        "norm_user_text_rendering_note": str(prompt_bundle.get("norm_user_text_rendering_note", "")),
        "evidence_card_fields": list(prompt_bundle.get("evidence_card_fields", [])),
        "component_prompt_roles": dict(prompt_bundle.get("component_prompt_roles", {})),
        "structured_component_schema": dict(prompt_bundle.get("structured_component_schema", {})),
        "embedding_encoder_tag": embedding_encoder_tag,
        "tweet_source_mode_requested": tweet_source_mode_requested,
        "tweet_source_mode_effective": tweet_source_mode_effective,
        "dgp_neighbor_summary_k": prompt_bundle.get("dgp_neighbor_summary_k", {}),
        "dgp_v2_directions": list(prompt_bundle.get("dgp_v2_directions", [])),
        "dgp_v2_summary_language": str(prompt_bundle.get("dgp_v2_summary_language", "")),
        "dgp_v2_empty_context_policy": str(prompt_bundle.get("dgp_v2_empty_context_policy", "")),
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
    elif prompt_bundle["prompt_family"] in {"prompt_expert_bundle_v2", "prompt_expert_bundle_v3"}:
        for scalar_key, scalar_values in prompt_bundle["scalar_features"].items():
            payload[scalar_key] = _scatter_selected_tensor_to_full_graph(
                torch.tensor(scalar_values, dtype=torch.float32),
                target_index_tensor,
                full_node_count,
            )
    write_torch(output_path, payload)
    component_cache_paths = {}
    if prompt_bundle["prompt_family"] in {
        "prompt_expert_bundle_v1",
        "prompt_expert_bundle_v2",
        "prompt_expert_bundle_v3",
        "prompt_expert_bundle_center_induced_v1",
    }:
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
        if prompt_family_version in {"v2", "v3"} or prompt_bundle["prompt_family"] in {"dgp_predictor_v2"}:
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
        "prompt_family_version": str(prompt_bundle.get("prompt_family_version", prompt_family_version)),
        "residual_prompt_variant": str(prompt_bundle.get("residual_prompt_variant", "")),
        "dgp_prompt_variant": str(prompt_bundle.get("dgp_prompt_variant", "")),
        "dgp_neighbor_summary_k": prompt_bundle.get("dgp_neighbor_summary_k", {}),
        "dgp_v2_directions": list(prompt_bundle.get("dgp_v2_directions", [])),
        "dgp_v2_summary_language": str(prompt_bundle.get("dgp_v2_summary_language", "")),
        "dgp_v2_empty_context_policy": str(prompt_bundle.get("dgp_v2_empty_context_policy", "")),
        "residual_base_outputs_path": str(prompt_bundle.get("base_outputs_path", "")),
        "residual_base_outputs_row_count": int(prompt_bundle.get("base_outputs_row_count", 0) or 0),
        "residual_base_output_keys": list(prompt_bundle.get("base_output_keys", [])),
        "residual_include_neighbor_base_distribution": bool(
            prompt_bundle.get("include_neighbor_base_distribution", False)
        ),
        "residual_class_names": list(prompt_bundle.get("class_names", [])),
        "evidence_schema": str(prompt_bundle.get("evidence_schema", "summary")),
        "norm_user_text_rendering": str(prompt_bundle.get("norm_user_text_rendering", "")),
        "norm_user_text_rendering_note": str(prompt_bundle.get("norm_user_text_rendering_note", "")),
        "evidence_card_fields": list(prompt_bundle.get("evidence_card_fields", [])),
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
        "explain_prompt_style": str(prompt_bundle.get("explain_prompt_style", "default")),
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
            "residual_audit_v1 emits a routed-node correction-utility prompt cache that treats frozen SimTeG "
            "predictions as optional, fallible hypotheses and never inserts dataset labels into prompt text. "
            "dgp_predictor_v1 emits a DGP-style dual-granularity bot/human predictor prompt cache with fine "
            "target evidence and coarse ranked following/follower neighbor context for Qwen PEFT predictor "
            "and Qwen-embedding-plus-MLP comparisons. "
            "v1 preserves the older mixed embedding/explanation behavior, while v2 upgrades expert prompts to an explanation-first pipeline with encoder-aware naming and pooling. "
            "v3 keeps the v2 explanation-first component contract but asks the explain model for source-grounded structured evidence cards without final labels, probabilities, confidence scores, recommendations, or model-correction instructions. "
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
