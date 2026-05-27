"""Precompute Qwen prompt embeddings for GLANCE-style, relation-aware, and expert views.

This helper builds per-node prompt texts, optionally generates ego explanations,
encodes them with Qwen embeddings, applies last-token pooling plus optional
l2 normalization, and stores a prompt cache under a stable ``embeddings`` key
for downstream graph_detector_prepare and strict GLANCE runs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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
NEIGHBOR_SAMPLING_CHOICES = ("auto", "uniform_random", "directional_heuristic")
EXPERT_COMPONENT_NAMES = ("ego", "graph_following", "graph_follower", "tweet", "conflict")
EXPERT_SCALAR_KEYS = (
    "count_following",
    "count_follower",
    "has_following",
    "has_follower",
    "rt_ratio",
    "url_ratio",
    "hashtag_ratio",
)


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum().item() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _load_texts(dataset_path: Path, text_path: Path | None):
    path = Path(text_path) if text_path else dataset_path / "norm_user_text.json"
    texts = read_json(path, default=None)
    if texts is None:
        raise FileNotFoundError(f"Could not read text cache: {path}")
    if not isinstance(texts, list):
        raise ValueError(f"Expected a list of node texts in {path}, got {type(texts).__name__}.")
    return texts, path


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


def _parse_norm_user_record(raw_text, node_id):
    metadata_text, description_text, tweets_text = _split_norm_user_text(raw_text)
    metadata_fields = [item.strip() for item in metadata_text.split(" </s> ")] if metadata_text else []
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
        "protected": bool(_parse_bool(protected)) if str(protected).strip() else False,
        "verified": bool(_parse_bool(verified)) if str(verified).strip() else False,
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
            f"The bio is {_truncate_chars(record['bio'], 160) or 'missing'}, which is a useful cue for downstream bot detection.",
        ]
    )


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
        "pytorch_model.bin",
    )
    return any((path / marker).exists() for marker in marker_files)


def _resolve_local_pretrained_source(model_source):
    text = str(model_source or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate)
    normalized = text.replace("\\", "/").strip("/")
    if not normalized or "/" not in normalized:
        return None
    cache_dir = _hf_cache_root() / f"models--{normalized.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    ordered = []
    main_ref = cache_dir / "refs" / "main"
    if main_ref.is_file():
        ref_name = main_ref.read_text(encoding="utf-8").strip()
        if ref_name:
            ordered.append(snapshots_dir / ref_name)
    ordered.extend(sorted(snapshots_dir.iterdir(), key=lambda item: item.name, reverse=True))
    seen = set()
    for snapshot_dir in ordered:
        snapshot_dir = Path(snapshot_dir)
        snapshot_text = str(snapshot_dir)
        if snapshot_text in seen:
            continue
        seen.add(snapshot_text)
        if _looks_like_pretrained_dir(snapshot_dir):
            return snapshot_text
    return None


def _require_local_pretrained_source(model_source, *, model_role):
    resolved = _resolve_local_pretrained_source(model_source)
    if resolved:
        return resolved
    raise FileNotFoundError(
        f"{model_role} source {model_source!r} is not available as a local path or local HuggingFace cache snapshot. "
        "Precompute runs in offline-first mode; provide a local snapshot path or pre-cache the model before running."
    )


def _fallback_explanation_text(row):
    text = str(row.get("fallback_explanation", "") or "").strip()
    if text:
        return text
    profile_card = row.get("profile_card", "")
    if profile_card:
        return " ".join(
            [
                "The profile provides limited generated explanation output.",
                _compact_whitespace(profile_card)[:320],
            ]
        ).strip()
    return "The profile provides limited generated explanation output."


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


def _build_glance_prompt_bundle(texts, edge_index, edge_type, cap, seed, classes):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    prompts = {"ego": [], "hop1": [], "hop2": []}
    counts = {"ego_nodes": [], "hop1_nodes": [], "hop2_nodes": []}
    for node_id, raw_text in enumerate(texts):
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


def _build_relation_aware_prompt_bundle(texts, edge_index, edge_type, following_quota, follower_quota, classes):
    num_nodes = len(texts)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    prompts = {"ego": [], "directional_1hop": []}
    counts = {
        "ego_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "directional_1hop_nodes": [],
    }
    for node_id, raw_text in enumerate(texts):
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


def _expert_output_name(prompt_mode):
    names = {
        "expert_ego": "glance_prompt_expert_ego_qwen3_embed.pt",
        "expert_graph_following": "glance_prompt_expert_graph_following_qwen3_embed.pt",
        "expert_graph_follower": "glance_prompt_expert_graph_follower_qwen3_embed.pt",
        "expert_tweet": "glance_prompt_expert_tweet_qwen3_embed.pt",
        "expert_conflict": "glance_prompt_expert_conflict_qwen3_embed.pt",
        "expert_concat_v1": "glance_prompt_expert_concat_v1_qwen3_embed.pt",
    }
    return names[prompt_mode]


def _default_output_path(dataset_path: Path, prompt_mode: str):
    if prompt_mode == "glance_concat_ego_hop1_hop2":
        return dataset_path / DEFAULT_OUTPUT_NAME
    if prompt_mode in EXPERT_PROMPT_MODES:
        return dataset_path / _expert_output_name(prompt_mode)
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


def _conflict_prompt(record, tweet_stats, following_records, follower_records, count_following, count_follower):
    flags = _conflict_flags(record, tweet_stats, count_following, count_follower)
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


def _ego_explain_generation_prompt(record):
    system = "You analyze Twitter accounts for bot detection."
    user = "\n".join(
        [
            "Given the profile card below, write a short explanation covering identity consistency, social reach, activity pattern, suspicious cues, and an overall judgment tendency.",
            "Do not output a label only; output 4-6 evidence-grounded sentences.",
            "PROFILE_CARD:",
            _profile_card(record, brief=False),
        ]
    )
    return {"system": system, "user": user}


def _ego_embedding_prompt(explanation_text):
    explanation_text = _compact_whitespace(explanation_text)
    return (
        "Instruct: Encode the profile-based bot-detection explanation for downstream classification.\n"
        f"Query: [EXPLANATION] {explanation_text} </END>"
    )


def _resolve_expert_prompt_bundle(args, records, edge_index, edge_type):
    num_nodes = len(records)
    context = _build_graph_context(edge_index, edge_type, num_nodes)
    mode = str(args.prompt_mode)
    selected_components = {
        "expert_ego": ["ego"],
        "expert_graph_following": ["graph_following"],
        "expert_graph_follower": ["graph_follower"],
        "expert_tweet": ["tweet"],
        "expert_conflict": ["conflict"],
        "expert_concat_v1": list(EXPERT_COMPONENT_NAMES),
    }[mode]
    prompt_components = {name: [] for name in selected_components if name != "ego"}
    counts = {
        "ego_nodes": [],
        "following_nodes": [],
        "follower_nodes": [],
        "tweet_nodes": [],
        "conflict_nodes": [],
    }
    scalar_features = {key: [] for key in EXPERT_SCALAR_KEYS}
    prompt_rows = []
    ego_generation_rows = []

    for node_id, record in enumerate(records):
        following_candidates = _rank_directional_candidates(node_id, context["following"][node_id], [item["raw_text"] for item in records], context)
        follower_candidates = _rank_directional_candidates(node_id, context["follower"][node_id], [item["raw_text"] for item in records], context)
        following_ids = following_candidates[: int(args.following_quota)]
        follower_ids = follower_candidates[: int(args.follower_quota)]
        following_records = [records[idx] for idx in following_ids]
        follower_records = [records[idx] for idx in follower_ids]
        reciprocal_count = len(set(context["following"][node_id]).intersection(set(context["follower"][node_id])))
        tweet_stats = _tweet_stats(record)
        scalar_features["count_following"].append(float(len(set(context["following"][node_id]))))
        scalar_features["count_follower"].append(float(len(set(context["follower"][node_id]))))
        scalar_features["has_following"].append(float(1 if context["following"][node_id] else 0))
        scalar_features["has_follower"].append(float(1 if context["follower"][node_id] else 0))
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

        row = {
            "node_id": int(node_id),
            "selected_components": list(selected_components),
            "profile_brief": _profile_card(record, brief=True),
            "tweet_summary": _tweet_behavior_summary(record, tweet_stats),
        }

        if "ego" in selected_components:
            generation_prompt = _ego_explain_generation_prompt(record)
            ego_generation_rows.append(
                {
                    "node_id": int(node_id),
                    "system": generation_prompt["system"],
                    "user": generation_prompt["user"],
                    "profile_card": _profile_card(record, brief=False),
                    "fallback_explanation": _deterministic_ego_explain_fallback(record),
                }
            )
            row["ego_generation_prompt"] = generation_prompt["user"]

        if "graph_following" in selected_components:
            prompt_components["graph_following"].append(
                _graph_prompt(
                    "following",
                    record,
                    following_records,
                    total_count=len(set(context["following"][node_id])),
                    reciprocal_count=reciprocal_count,
                )
            )
            row["graph_following_prompt"] = prompt_components["graph_following"][-1]

        if "graph_follower" in selected_components:
            prompt_components["graph_follower"].append(
                _graph_prompt(
                    "follower",
                    record,
                    follower_records,
                    total_count=len(set(context["follower"][node_id])),
                    reciprocal_count=reciprocal_count,
                )
            )
            row["graph_follower_prompt"] = prompt_components["graph_follower"][-1]

        if "tweet" in selected_components:
            prompt_components["tweet"].append(_tweet_prompt(record, tweet_stats))
            row["tweet_prompt"] = prompt_components["tweet"][-1]

        if "conflict" in selected_components:
            conflict_prompt, flags = _conflict_prompt(
                record,
                tweet_stats,
                following_records,
                follower_records,
                count_following=len(set(context["following"][node_id])),
                count_follower=len(set(context["follower"][node_id])),
            )
            prompt_components["conflict"].append(conflict_prompt)
            row["conflict_prompt"] = conflict_prompt
            row["conflict_flags"] = flags

        prompt_rows.append(row)

    return {
        "prompt_family": "prompt_expert_bundle_v1",
        "prompt_style": "expert_prompt_bundle_v1",
        "prompt_components": prompt_components,
        "ego_generation_rows": ego_generation_rows,
        "component_max_length_group": {
            "ego": "ego",
            "graph_following": "hop",
            "graph_follower": "hop",
            "tweet": "hop",
            "conflict": "hop",
        },
        "counts": counts,
        "neighbor_sample_policy": "directional_heuristic_fixed_quota",
        "directional_quota": {"following": int(args.following_quota), "follower": int(args.follower_quota)},
        "component_order": {mode_name: list(component_names) for mode_name, component_names in {
            "expert_ego": ["ego"],
            "expert_graph_following": ["graph_following"],
            "expert_graph_follower": ["graph_follower"],
            "expert_tweet": ["tweet"],
            "expert_conflict": ["conflict"],
            "expert_concat_v1": list(EXPERT_COMPONENT_NAMES),
        }.items()},
        "selected_components": list(selected_components),
        "scalar_features": scalar_features,
        "prompt_rows": prompt_rows,
        "semantic_view_mode": "prompt_expert_bundle_v1",
    }


def _resolve_prompt_bundle(args, texts, edge_index, edge_type, classes):
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
        )
    elif mode.startswith("expert_"):
        if policy not in {"auto", "directional_heuristic"}:
            raise ValueError(f"{mode} only supports neighbor sampling policy 'directional_heuristic' or 'auto'.")
        records = [_parse_norm_user_record(raw_text, idx) for idx, raw_text in enumerate(texts)]
        bundle = _resolve_expert_prompt_bundle(args, records, edge_index, edge_type)
    else:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    selected_components = bundle["component_order"].get(mode)
    if not selected_components:
        raise ValueError(f"Prompt bundle did not define selected components for mode {mode}.")
    bundle["prompt_mode"] = mode
    bundle["selected_components"] = list(selected_components)
    bundle["neighbor_sampling_policy_requested"] = policy
    return bundle


def _device_from_args(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_embedding_model(args, device):
    from transformers import AutoModel, AutoTokenizer

    model_source = _require_local_pretrained_source(args.model_path, model_role="Embedding model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def _encode_texts(model, tokenizer, texts, device, batch_size, max_length, normalize):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=torch.float32)


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
            f"Prompt mode {args.prompt_mode} requires --explain_model_path for ego explanation generation."
        )
    model_source = _require_local_pretrained_source(args.explain_model_path, model_role="Explain model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=bool(args.explain_trust_remote_code),
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=bool(args.explain_trust_remote_code),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def _generate_explanations(args, prompt_bundle, device):
    generation_rows = list(prompt_bundle.get("ego_generation_rows", []))
    if not generation_rows:
        return [], {"generated_count": 0, "fallback_count": 0}
    if not args.explain_model_path:
        explanations = _deterministic_explanations(generation_rows)
        return explanations, {
            "generated_count": 0,
            "fallback_count": len(explanations),
            "generation_mode": "deterministic_fallback",
            "fallback_reason": "missing_explain_model_path",
            "explain_model_path": "",
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
        }
    try:
        tokenizer, model = _load_generation_model(args, device)
    except Exception as exc:
        explanations = _deterministic_explanations(generation_rows)
        return explanations, {
            "generated_count": 0,
            "fallback_count": len(explanations),
            "generation_mode": "deterministic_fallback",
            "fallback_reason": "local_explain_model_unavailable",
            "explain_model_error": str(exc),
            "explain_model_path": str(args.explain_model_path),
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
        }
    prompts = [
        _format_chat_prompt(tokenizer, row["system"], row["user"])
        for row in generation_rows
    ]
    explanations = []
    fallback_count = 0
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), int(args.explain_batch_size)):
                prompt_batch = prompts[start : start + int(args.explain_batch_size)]
                batch_rows = generation_rows[start : start + int(args.explain_batch_size)]
                batch = tokenizer(
                    prompt_batch,
                    padding=True,
                    truncation=True,
                    max_length=int(args.explain_max_input_length),
                    return_tensors="pt",
                )
                input_lengths = batch["attention_mask"].sum(dim=1)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model.generate(
                    **batch,
                    max_new_tokens=int(args.explain_max_new_tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for row_idx, generated_ids in enumerate(outputs):
                    prompt_length = int(input_lengths[row_idx].item())
                    continuation = generated_ids[prompt_length:]
                    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                    if not text:
                        fallback_count += 1
                        text = _fallback_explanation_text(batch_rows[row_idx])
                    explanations.append(text)
    except Exception as exc:
        explanations = _deterministic_explanations(generation_rows)
        return explanations, {
            "generated_count": 0,
            "fallback_count": len(explanations),
            "generation_mode": "deterministic_fallback",
            "fallback_reason": "generation_failed",
            "explain_model_error": str(exc),
            "explain_model_path": str(args.explain_model_path),
            "explain_batch_size": int(args.explain_batch_size),
            "explain_max_input_length": int(args.explain_max_input_length),
            "explain_max_new_tokens": int(args.explain_max_new_tokens),
        }
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "generated_count": len(explanations),
        "fallback_count": int(fallback_count),
        "generation_mode": "llm_generation",
        "explain_model_path": str(args.explain_model_path),
        "explain_batch_size": int(args.explain_batch_size),
        "explain_max_input_length": int(args.explain_max_input_length),
        "explain_max_new_tokens": int(args.explain_max_new_tokens),
    }
    return explanations, summary


def _attach_ego_explanations(prompt_bundle, explanations):
    generation_rows = prompt_bundle.get("ego_generation_rows", [])
    if generation_rows and len(explanations) != len(generation_rows):
        raise ValueError(
            "Generated ego explanation count does not match the number of ego generation prompts."
        )
    if "ego" in prompt_bundle["selected_components"]:
        prompt_bundle["prompt_components"]["ego"] = [_ego_embedding_prompt(item) for item in explanations]
        for row, explanation in zip(prompt_bundle["prompt_rows"], explanations):
            row["ego_explanation"] = explanation
            row["ego_embedding_prompt"] = _ego_embedding_prompt(explanation)
    return prompt_bundle


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute GLANCE-style, relation-aware, or expert Qwen prompt embeddings.")
    parser.add_argument("--dataset", type=str, default="TwiBot-20")
    parser.add_argument("--text_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--model_path", type=str, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_mode", choices=PROMPT_MODE_CHOICES, default="glance_concat_ego_hop1_hop2")
    parser.add_argument("--neighbor_sampling_policy", choices=NEIGHBOR_SAMPLING_CHOICES, default="auto")
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
    parser.add_argument("--explain_batch_size", type=int, default=1)
    parser.add_argument("--explain_max_input_length", type=int, default=2048)
    parser.add_argument("--explain_max_new_tokens", type=int, default=192)
    return parser


def run(args):
    dataset_path = resolve_dataset_path(args.dataset)
    texts, resolved_text_path = _load_texts(dataset_path, args.text_path)
    if args.limit and int(args.limit) > 0:
        texts = texts[: int(args.limit)]
    if not texts:
        raise ValueError("No node texts found for Qwen prompt precompute.")

    labels_path = dataset_path / "labels.pt"
    labels = torch.load(labels_path, map_location="cpu")
    if torch.is_tensor(labels) and labels.dim() > 1:
        class_count = int(labels.shape[1])
        classes = ["human", "bot"] if class_count == 2 else [f"class_{idx}" for idx in range(class_count)]
    else:
        classes = sorted(int(x) for x in torch.unique(torch.as_tensor(labels)).tolist())

    edge_index = torch.load(dataset_path / "edge_index.pt", map_location="cpu")
    edge_type = torch.load(dataset_path / "edge_type.pt", map_location="cpu")
    prompt_bundle = _resolve_prompt_bundle(args, texts, edge_index, edge_type, classes)

    output_path = Path(args.output_path) if args.output_path else _default_output_path(dataset_path, str(args.prompt_mode))
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or a different --output_path.")
    ensure_dir(output_path.parent)

    explain_summary = None
    explain_rows = []
    device = _device_from_args(str(args.device))
    if str(args.prompt_mode) in {"expert_ego", "expert_concat_v1"}:
        explain_rows, explain_summary = _generate_explanations(args, prompt_bundle, device)
        prompt_bundle = _attach_ego_explanations(prompt_bundle, explain_rows)

    tokenizer, model = _load_embedding_model(args, device)
    encoded = {}
    for component_name, component_prompts in prompt_bundle["prompt_components"].items():
        if not component_prompts:
            continue
        length_group = prompt_bundle["component_max_length_group"].get(component_name, "hop")
        max_length = int(args.max_length_ego) if length_group == "ego" else int(args.max_length_hop)
        encoded[component_name] = _encode_texts(
            model,
            tokenizer,
            component_prompts,
            device,
            int(args.batch_size),
            max_length,
            bool(args.normalize),
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
    payload = {
        "embeddings": selected_embeddings.to(save_dtype).contiguous(),
        "semantic_view_mode": prompt_bundle.get("semantic_view_mode", str(args.prompt_mode)),
        "active_components": list(selected_components),
    }
    for component_name, component_tensor in encoded.items():
        payload[component_name] = component_tensor.to(save_dtype).contiguous()
    if prompt_bundle["prompt_family"] == "prompt_expert_bundle_v1":
        for scalar_key, scalar_values in prompt_bundle["scalar_features"].items():
            payload[scalar_key] = torch.tensor(scalar_values, dtype=torch.float32)
    write_torch(output_path, payload)

    tensor_hashes = {"embeddings": tensor_sha256(payload["embeddings"])}
    for component_name in encoded:
        tensor_hashes[component_name] = tensor_sha256(payload[component_name])
    for scalar_key in prompt_bundle.get("scalar_features", {}):
        tensor_hashes[scalar_key] = tensor_sha256(payload[scalar_key])

    component_budget = {}
    for component_name, group in prompt_bundle["component_max_length_group"].items():
        component_budget[component_name] = int(args.max_length_ego) if group == "ego" else int(args.max_length_hop)

    prompt_sidecar_path = output_path.with_name(f"{output_path.stem}_prompts.jsonl")
    _write_jsonl(prompt_sidecar_path, prompt_bundle.get("prompt_rows", []))
    explain_sidecar_path = None
    if explain_rows:
        explain_sidecar_path = output_path.with_name("glance_prompt_expert_ego_explain.jsonl")
        _write_jsonl(
            explain_sidecar_path,
            [
                {
                    "node_id": int(row["node_id"]),
                    "explanation": explanation,
                }
                for row, explanation in zip(prompt_bundle["ego_generation_rows"], explain_rows)
            ],
        )

    manifest = {
        "status": "completed",
        "dataset": str(args.dataset),
        "dataset_path": str(dataset_path),
        "text_path": str(resolved_text_path),
        "output_path": str(output_path),
        "model_path": str(args.model_path),
        "trust_remote_code": bool(args.trust_remote_code),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "prompt_mode": str(args.prompt_mode),
        "prompt_family": prompt_bundle["prompt_family"],
        "prompt_style": prompt_bundle["prompt_style"],
        "neighbor_sampling_policy": prompt_bundle["neighbor_sample_policy"],
        "neighbor_sampling_policy_requested": prompt_bundle["neighbor_sampling_policy_requested"],
        "neighbor_cap": int(args.neighbor_cap),
        "following_quota": int(args.following_quota),
        "follower_quota": int(args.follower_quota),
        "max_length_ego": int(args.max_length_ego),
        "max_length_hop": int(args.max_length_hop),
        "component_prompt_token_budget": component_budget,
        "normalize": bool(args.normalize),
        "save_dtype": str(args.save_dtype),
        "limit": int(args.limit),
        "semantic_mode": str(args.prompt_mode),
        "semantic_view_mode": prompt_bundle.get("semantic_view_mode", str(args.prompt_mode)),
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
        "prompt_sidecar_path": str(prompt_sidecar_path),
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
            "expert_ego/expert_concat_v1 now stay offline-first: they reuse a local explain-model snapshot when available "
            "and otherwise fall back to deterministic profile-card explanations instead of blocking on HuggingFace downloads."
        ),
    }
    if explain_summary is not None:
        manifest["ego_explain_generation"] = explain_summary
    write_json(output_path.with_name(f"{output_path.stem}_manifest.json"), manifest)
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
    run(args)


if __name__ == "__main__":
    main()
