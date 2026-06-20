from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import ijson


FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "data"
PROJECT_ROOT = FIG_DIR.parents[2]
TWIBOT_ROOT = PROJECT_ROOT / "datasets" / "TwiBot-20"
SELECTED_CSV = DATA_DIR / "case_study_selected_nodes.csv"
SUPPORT_CSV = DATA_DIR / "case_study_support_nodes.csv"
OUTPUT_JSON = DATA_DIR / "casestudy_node11654_evidence.json"

TARGET_SEED = "2"
TARGET_NODE = 11654
TARGET_CASE_TYPE = "routed_correction"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label_name(value: str | int) -> str:
    return "Bot" if int(value) == 1 else "Human"


def _shorten(text: str, limit: int = 130) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _load_ordered_user_ids() -> list[str]:
    split = json.loads((TWIBOT_ROOT / "split_new.json").read_text(encoding="utf-8"))
    return list(split["train"]) + list(split["dev"]) + list(split["test"])


def _stream_profiles(user_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with (TWIBOT_ROOT / "node_new.json").open("rb") as handle:
        for user_id, obj in ijson.kvitems(handle, ""):
            if user_id in user_ids:
                found[user_id] = obj
                if len(found) == len(user_ids):
                    break
    missing = sorted(user_ids - set(found))
    if missing:
        raise KeyError(f"Missing user ids in node_new.json: {missing}")
    return found


def _stream_norm_text(target_index: int) -> str:
    with (TWIBOT_ROOT / "norm_user_text_new.json").open("rb") as handle:
        for idx, item in enumerate(ijson.items(handle, "item")):
            if idx == target_index:
                return str(item)
            if idx > target_index:
                break
    raise IndexError(f"Target index {target_index} not found in norm_user_text_new.json")


def _extract_tweets(norm_text: str, max_items: int = 6) -> list[str]:
    marker = " TWEET:"
    if marker not in norm_text:
        return []
    tweet_part = norm_text.split(marker, 1)[1]
    tweets = [_shorten(tweet, 112) for tweet in tweet_part.split(" </s> ")]
    return [tweet for tweet in tweets if tweet][:max_items]


def _profile_summary(index: int, user_id: str, profile: dict[str, Any], label: str, **extra: str) -> dict[str, str]:
    metrics = profile.get("public_metrics") or {}
    row = {
        "node_index": str(index),
        "user_id": user_id,
        "username": str(profile.get("username") or "").strip(),
        "name": str(profile.get("name") or "").strip(),
        "label": label,
        "label_name": _label_name(label),
        "followers_count": str(metrics.get("followers_count", "")),
        "following_count": str(metrics.get("following_count", "")),
        "tweet_count": str(metrics.get("tweet_count", "")),
        "listed_count": str(metrics.get("listed_count", "")),
        "verified": str(profile.get("verified", "")).strip(),
    }
    row.update(extra)
    return row


def build_evidence() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selected_rows = _read_csv(SELECTED_CSV)
    support_rows = _read_csv(SUPPORT_CSV)
    selected = [
        row
        for row in selected_rows
        if row["case_type"] == TARGET_CASE_TYPE
        and row["seed"] == TARGET_SEED
        and int(row["node_id"]) == TARGET_NODE
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected row for seed={TARGET_SEED}, node={TARGET_NODE}; found {len(selected)}")
    row = selected[0]

    support_for_case = [
        item
        for item in support_rows
        if item["case_type"] == TARGET_CASE_TYPE
        and item["seed"] == TARGET_SEED
        and int(item["node_id"]) == TARGET_NODE
    ]
    if len(support_for_case) != 8:
        raise ValueError(f"Expected 8 support nodes; found {len(support_for_case)}")

    assert row["gold"] == "1", row
    assert row["low_pred"] == "0", row
    assert row["high_pred"] == "1", row
    assert row["final_pred"] == "1", row
    assert row["routed_mask"] == "True", row
    assert [item["support_label"] for item in support_for_case] == ["1", "1", "1", "1", "1", "0", "1", "1"]

    ordered_ids = _load_ordered_user_ids()
    target_user_id = ordered_ids[TARGET_NODE]
    relation_indices = [int(value) for value in row["relation_neighbor_ids"].split()]
    relation_labels = row["relation_neighbor_labels"].split()
    support_indices = [int(item["support_id"]) for item in support_for_case]
    target_indices = [TARGET_NODE] + relation_indices + support_indices
    user_ids = {ordered_ids[idx] for idx in target_indices}
    profiles = _stream_profiles(user_ids)
    norm_text = _stream_norm_text(TARGET_NODE)
    target_profile = profiles[target_user_id]
    target_metrics = target_profile.get("public_metrics") or {}

    if target_user_id != "u113490630":
        raise AssertionError(f"Expected target user id u113490630; got {target_user_id}")
    if str(target_profile.get("username", "")).strip() != "PeterBotte":
        raise AssertionError(f"Expected username PeterBotte; got {target_profile.get('username')}")

    relation_neighbors = []
    for idx, label in zip(relation_indices, relation_labels):
        user_id = ordered_ids[idx]
        relation_neighbors.append(
            _profile_summary(idx, user_id, profiles[user_id], label, relation_type="relation")
        )

    support_neighbors = []
    for item in support_for_case:
        idx = int(item["support_id"])
        user_id = ordered_ids[idx]
        support_neighbors.append(
            _profile_summary(
                idx,
                user_id,
                profiles[user_id],
                item["support_label"],
                rank=item["rank"],
                similarity=item["support_similarity"],
                relation_type="knn_support",
            )
        )

    evidence = {
        "case": {
            "case_type": TARGET_CASE_TYPE,
            "seed": int(TARGET_SEED),
            "node_index": TARGET_NODE,
            "gold": row["gold"],
            "gold_name": _label_name(row["gold"]),
            "risk_score": round(float(row["risk_score"]), 4),
            "routed": True,
            "relation_bot_ratio": round(float(row["relation_bot_ratio"]), 3),
            "support_bot_ratio": round(float(row["support_bot_ratio"]), 3),
            "knn_prediction_disagreement": round(float(row["knn_prediction_disagreement"]), 3),
        },
        "target": {
            "user_id": target_user_id,
            "username": str(target_profile.get("username") or "").strip(),
            "name": str(target_profile.get("name") or "").strip(),
            "description": _shorten(str(target_profile.get("description") or ""), 170),
            "created_at": str(target_profile.get("created_at") or "").strip(),
            "verified": str(target_profile.get("verified", "")).strip(),
            "protected": str(target_profile.get("protected", "")).strip(),
            "followers_count": int(target_metrics.get("followers_count", 0)),
            "following_count": int(target_metrics.get("following_count", 0)),
            "tweet_count": int(target_metrics.get("tweet_count", 0)),
            "listed_count": int(target_metrics.get("listed_count", 0)),
            "tweets": _extract_tweets(norm_text),
        },
        "relation_neighbors": relation_neighbors,
        "support_neighbors": support_neighbors,
        "branch_evidence": [
            {
                "name": "Low-order/base",
                "short_name": "Low",
                "prediction": _label_name(row["low_pred"]),
                "prediction_value": int(row["low_pred"]),
                "true_confidence": round(float(row["low_conf_true"]), 4),
                "margin_true_minus_other": round(float(row["low_margin_true_minus_other"]), 4),
                "correct": row["low_pred"] == row["gold"],
            },
            {
                "name": "High-order branch",
                "short_name": "High",
                "prediction": _label_name(row["high_pred"]),
                "prediction_value": int(row["high_pred"]),
                "true_confidence": round(float(row["high_conf_true"]), 4),
                "margin_true_minus_other": round(float(row["high_margin_true_minus_other"]), 4),
                "correct": row["high_pred"] == row["gold"],
            },
            {
                "name": "All-node residual",
                "short_name": "All-node",
                "prediction": _label_name(row["all_nodes_pred"]),
                "prediction_value": int(row["all_nodes_pred"]),
                "true_confidence": round(float(row["all_nodes_conf_true"]), 4),
                "margin_true_minus_other": round(float(row["all_nodes_margin_true_minus_other"]), 4),
                "correct": row["all_nodes_pred"] == row["gold"],
            },
            {
                "name": "Routed-only final",
                "short_name": "Final",
                "prediction": _label_name(row["final_pred"]),
                "prediction_value": int(row["final_pred"]),
                "true_confidence": round(float(row["final_conf_true"]), 4),
                "margin_true_minus_other": round(float(row["final_margin_true_minus_other"]), 4),
                "correct": row["final_pred"] == row["gold"],
            },
        ],
        "interpretation": [
            "Low-order/base evidence predicts Human for a Bot-labeled routed node.",
            "KNN support is strongly Bot-dominant, providing high-order evidence for correction.",
            "The routed-only final branch keeps the high-order correction on the selected hard node.",
        ],
        "sources": {
            "selected_case_csv": str(SELECTED_CSV),
            "support_case_csv": str(SUPPORT_CSV),
            "node_json": str(TWIBOT_ROOT / "node_new.json"),
            "norm_text_json": str(TWIBOT_ROOT / "norm_user_text_new.json"),
            "routed_json": row["routed_json"],
            "risk_manifest": row["risk_manifest"],
        },
    }

    OUTPUT_JSON.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    evidence = build_evidence()
    print(f"Wrote {OUTPUT_JSON}")
    print(
        "Target:",
        evidence["target"]["user_id"],
        evidence["target"]["username"],
        "tweets=",
        len(evidence["target"]["tweets"]),
    )


if __name__ == "__main__":
    main()
