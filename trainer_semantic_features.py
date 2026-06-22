import math
import re
from pathlib import Path

import torch

from utils import safe_torch_load


def _as_long_cpu_tensor(idx):
    if idx is None:
        return torch.empty(0, dtype=torch.long)
    if torch.is_tensor(idx):
        return idx.detach().cpu().long().reshape(-1)
    return torch.tensor(idx, dtype=torch.long).reshape(-1)


def _none_like_text(value):
    token = str(value or "").strip().lower()
    return token in {"", "none", "null", "nan", "unknown"}


def _parse_bool_feature(value):
    token = str(value or "").strip().lower()
    if token in {"true", "1", "yes", "y"}:
        return 1.0
    if token in {"false", "0", "no", "n", "none", ""}:
        return 0.0
    return 0.0


def _safe_float_feature(value):
    token = str(value or "").replace(",", " ")
    match = re.search(r"-?\d+(?:\.\d+)?", token)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


def _log1p_feature(value):
    return float(math.log1p(max(float(value), 0.0)))


def _char_ratio_feature(text, predicate):
    text = str(text or "")
    if not text:
        return 0.0
    return float(sum(1 for ch in text if predicate(ch))) / float(len(text))


def _split_norm_user_text(text):
    text = str(text or "")
    upper = text.upper()
    meta_anchor = "METADATA:"
    desc_anchor = "DESCRIPTION:"
    tweet_anchor = "TWEET:"
    meta_idx = upper.find(meta_anchor)
    desc_idx = upper.find(desc_anchor)
    tweet_idx = upper.find(tweet_anchor)
    metadata = ""
    description = ""
    tweets = ""
    if meta_idx >= 0:
        meta_start = meta_idx + len(meta_anchor)
        meta_end = desc_idx if desc_idx >= 0 else (tweet_idx if tweet_idx >= 0 else len(text))
        metadata = text[meta_start:meta_end].strip()
    if desc_idx >= 0:
        desc_start = desc_idx + len(desc_anchor)
        desc_end = tweet_idx if tweet_idx >= 0 else len(text)
        description = text[desc_start:desc_end].strip()
    if tweet_idx >= 0:
        tweets = text[tweet_idx + len(tweet_anchor) :].strip()
    metadata_fields = [item.strip() for item in metadata.split("</s>")]
    while len(metadata_fields) < 10:
        metadata_fields.append("")
    tweet_items = [item.strip() for item in tweets.split("</s>") if item.strip()]
    return metadata_fields, description, tweets, tweet_items


def _build_text_attribute_features(user_text, row_count):
    names = [
        "protected",
        "verified",
        "created_at_present",
        "location_present",
        "display_name_present",
        "screen_name_present",
        "bio_present",
        "followers_log1p",
        "following_log1p",
        "listed_log1p",
        "statuses_log1p",
        "followers_following_log_ratio",
        "listed_followers_log_ratio",
        "statuses_followers_log_ratio",
        "bio_char_len_log1p",
        "display_name_char_len_log1p",
        "screen_name_char_len_log1p",
        "screen_name_digit_ratio",
        "screen_name_underscore_ratio",
        "tweet_count_log1p",
        "tweet_char_len_log1p",
        "rt_ratio",
        "url_ratio",
        "hashtag_ratio",
        "mention_ratio",
        "emoji_ratio",
        "no_tweet_evidence",
    ]
    matrix = torch.zeros((int(row_count), len(names)), dtype=torch.float32)
    usable = min(int(row_count), len(user_text or []))
    for node_idx in range(usable):
        fields, description, tweets, tweet_items = _split_norm_user_text(user_text[node_idx])
        created_at, location, display_name, protected = fields[0], fields[1], fields[2], fields[3]
        followers = _safe_float_feature(fields[4])
        following = _safe_float_feature(fields[5])
        listed = _safe_float_feature(fields[6])
        statuses = _safe_float_feature(fields[7])
        screen_name = fields[8]
        verified = fields[9]
        tweet_count = max(len(tweet_items), 0)
        lower_tweets = [item.lower() for item in tweet_items]
        denom = float(max(tweet_count, 1))
        rt_count = sum(1 for item in lower_tweets if item.startswith("rt ") or " rt @user" in item or item.startswith("rt @user"))
        url_count = sum(1 for item in lower_tweets if "httpurl" in item or "http://" in item or "https://" in item)
        hashtag_count = sum(1 for item in lower_tweets if "#hashtag" in item or "#" in item)
        mention_count = sum(1 for item in lower_tweets if "@user" in item or "@" in item)
        emoji_count = sum(1 for item in lower_tweets if "emoji" in item)
        row = [
            _parse_bool_feature(protected),
            _parse_bool_feature(verified),
            0.0 if _none_like_text(created_at) else 1.0,
            0.0 if _none_like_text(location) else 1.0,
            0.0 if _none_like_text(display_name) else 1.0,
            0.0 if _none_like_text(screen_name) else 1.0,
            0.0 if _none_like_text(description) else 1.0,
            _log1p_feature(followers),
            _log1p_feature(following),
            _log1p_feature(listed),
            _log1p_feature(statuses),
            _log1p_feature(followers) - _log1p_feature(following),
            _log1p_feature(listed) - _log1p_feature(followers),
            _log1p_feature(statuses) - _log1p_feature(max(followers, 1.0)),
            _log1p_feature(len(str(description or ""))),
            _log1p_feature(len(str(display_name or ""))),
            _log1p_feature(len(str(screen_name or ""))),
            _char_ratio_feature(screen_name, lambda ch: ch.isdigit()),
            _char_ratio_feature(screen_name, lambda ch: ch == "_"),
            _log1p_feature(tweet_count),
            _log1p_feature(len(str(tweets or ""))),
            float(rt_count) / denom,
            float(url_count) / denom,
            float(hashtag_count) / denom,
            float(mention_count) / denom,
            float(emoji_count) / denom,
            1.0 if tweet_count == 0 else 0.0,
        ]
        matrix[node_idx] = torch.tensor(row, dtype=torch.float32)
    return matrix, names


def _resolve_gate_edge_tensors(data):
    if "edge_index" in data and "edge_type" in data:
        return data["edge_index"], data["edge_type"], "data"
    dataset_path = Path(data.get("dataset_path", ""))
    edge_index_path = dataset_path / "edge_index.pt"
    edge_type_path = dataset_path / "edge_type.pt"
    if edge_index_path.exists() and edge_type_path.exists():
        return safe_torch_load(edge_index_path, map_location="cpu"), safe_torch_load(edge_type_path, map_location="cpu"), "dataset_labeled_graph"
    return None, None, "missing"


def _build_graph_attribute_features(data, row_count):
    names = [
        "graph_following_log1p",
        "graph_follower_log1p",
        "graph_has_following",
        "graph_has_follower",
        "graph_total_degree_log1p",
        "graph_following_follower_log_ratio",
        "graph_reciprocal_ratio",
        "graph_neighbor_activity_log1p",
        "graph_isolated",
    ]
    matrix = torch.zeros((int(row_count), len(names)), dtype=torch.float32)
    edge_index, edge_type, source = _resolve_gate_edge_tensors(data)
    if edge_index is None or edge_type is None:
        return matrix, names, {"graph_attribute_source": source, "graph_attribute_available": False}
    edge_index = edge_index.detach().cpu().long()
    edge_type = edge_type.detach().cpu().long().reshape(-1)
    if edge_index.dim() != 2 or edge_index.shape[0] != 2 or edge_type.numel() != edge_index.shape[1]:
        return matrix, names, {"graph_attribute_source": source, "graph_attribute_available": False, "graph_attribute_error": "invalid_edge_shape"}
    src = edge_index[0].clamp_min(0)
    dst = edge_index[1].clamp_min(0)
    valid = (src < int(row_count)) & (dst < int(row_count))
    src = src[valid]
    dst = dst[valid]
    rel = edge_type[valid]
    following_mask = rel == 1
    follower_mask = rel == 0
    following = torch.bincount(src[following_mask], minlength=int(row_count)).float()
    follower = torch.bincount(dst[follower_mask], minlength=int(row_count)).float()
    out_all = torch.bincount(src, minlength=int(row_count)).float()
    in_all = torch.bincount(dst, minlength=int(row_count)).float()
    total = out_all + in_all
    neighbor_sum = torch.zeros(int(row_count), dtype=torch.float32)
    if src.numel():
        neighbor_total = total
        neighbor_sum.index_add_(0, src, neighbor_total[dst])
        neighbor_sum.index_add_(0, dst, neighbor_total[src])
    neighbor_mean = neighbor_sum / total.clamp_min(1.0)
    neighbor_sets = [set() for _ in range(int(row_count))]
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbor_sets[int(s)].add(int(d))
    reciprocal = torch.zeros(int(row_count), dtype=torch.float32)
    for node_idx, neighbors in enumerate(neighbor_sets):
        if not neighbors:
            continue
        reciprocal_count = sum(1 for nbr in neighbors if node_idx in neighbor_sets[nbr])
        reciprocal[node_idx] = float(reciprocal_count) / float(len(neighbors))
    matrix = torch.stack(
        [
            torch.log1p(following),
            torch.log1p(follower),
            (following > 0).float(),
            (follower > 0).float(),
            torch.log1p(total),
            torch.log1p(following) - torch.log1p(follower),
            reciprocal,
            torch.log1p(neighbor_mean),
            (total == 0).float(),
        ],
        dim=1,
    ).float()
    return matrix, names, {
        "graph_attribute_source": source,
        "graph_attribute_available": True,
        "edge_count_used": int(src.numel()),
    }


def _standardize_attribute_features(features, train_idx):
    train_idx = _as_long_cpu_tensor(train_idx)
    reference = features[train_idx] if train_idx.numel() else features
    mean = reference.mean(dim=0, keepdim=True)
    std = reference.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    normalized = ((features - mean) / std).clamp(-10.0, 10.0)
    return normalized.float(), {
        "normalization": "train_zscore_clamped_10",
        "mean": mean.reshape(-1).tolist(),
        "std": std.reshape(-1).tolist(),
    }


def _build_semantic_gate_node_attribute_features(data, train_idx, row_count):
    text_features, text_names = _build_text_attribute_features(data.get("user_text", []), row_count)
    graph_features, graph_names, graph_meta = _build_graph_attribute_features(data, row_count)
    raw = torch.cat([text_features, graph_features], dim=1).float()
    features, norm_meta = _standardize_attribute_features(raw, train_idx)
    names = list(text_names) + list(graph_names)
    return features, names, {
        "feature_source": "norm_user_text_plus_labeled_graph",
        "raw_feature_dim": int(raw.shape[1]),
        "feature_names": names,
        **graph_meta,
        **norm_meta,
    }
