"""Build support-extended TwiBot-20 artifacts without changing supervision splits."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from array import array
from pathlib import Path


SPECIAL_TOKENS = ["DESCRIPTION:", "METADATA:", "TWEET:", "@USER", "#HASHTAG", "HTTPURL", "EMOJI", "RT", "None"]
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNK_SIZE = 1 << 20


def _parse_args():
    parser = argparse.ArgumentParser(description="Build support-extended TwiBot-20 artifacts.")
    parser.add_argument("--dataset_root", type=Path, default=Path(__file__).resolve().parents[1] / "datasets" / "TwiBot-20")
    parser.add_argument("--mode", choices=("artifacts", "encode_support"), default="artifacts")
    parser.add_argument("--roberta_model_path", type=str, default=None)
    parser.add_argument("--roberta_checkpoint_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_torch(path: Path, payload):
    import torch

    torch.save(payload, path)


def _read_torch(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_roberta_backbone_checkpoint(model, checkpoint_path: Path):
    checkpoint = _read_torch(Path(checkpoint_path))
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload at {checkpoint_path}")
    backbone_state = {}
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        if key.startswith("LM."):
            backbone_state[key[len("LM."):]] = value
    if not backbone_state:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain any 'LM.' backbone weights.")
    incompatible = model.load_state_dict(backbone_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_key_count": len(backbone_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _canonical_user_id(raw_id):
    text = str(raw_id).strip()
    if not text:
        return ""
    if text.startswith("u") or text.startswith("t"):
        return text
    return f"u{text}"


def _resolve_official_edge_csv_path(dataset_root: Path):
    candidates = [
        dataset_root / "edge.csv",
        Path(r"F:\Twibot20\TwiBot-20-Format22\Twibot-20\edge.csv"),
        Path(r"F:\Twibot20\data\TwiBot-20\edge.csv"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not resolve official TwiBot-20 edge.csv. "
        "Checked dataset_root/edge.csv and known local Format22 paths."
    )


def _compact_text(value):
    if value is None:
        return "None"
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"\s+[+-]\d{4}\s+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", text)
    text = re.sub(r"(?<!\w)@\w+", "@USER", text)
    text = re.sub(r"(?<!\w)#\w+", "#HASHTAG", text)
    text = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
        " EMOJI ",
        text,
        flags=re.UNICODE,
    )
    text = " ".join(text.replace("\n", " ").split()).strip()
    return text or "None"


def _compact_field(value):
    if value is None:
        return "None"
    text = str(value).strip()
    return text if text else "None"


def _support_profile_to_text(record):
    profile = record.get("profile") or {}
    created_at = _compact_field(profile.get("created_at"))
    location = _compact_field(profile.get("location"))
    name = _compact_field(profile.get("name"))
    protected = _compact_field(profile.get("protected"))
    followers_count = _compact_field(profile.get("followers_count"))
    following_count = _compact_field(profile.get("friends_count") or profile.get("following_count"))
    listed_count = _compact_field(profile.get("listed_count"))
    statuses_count = _compact_field(profile.get("statuses_count") or profile.get("tweet_count"))
    screen_name = _compact_field(profile.get("screen_name") or profile.get("username"))
    verified = _compact_field(profile.get("verified"))
    description = _compact_text(profile.get("description"))

    tweets = record.get("tweet") or []
    if isinstance(tweets, list):
        tweet_texts = [_compact_text(tweet) for tweet in tweets if str(tweet).strip()]
    else:
        tweet_texts = [_compact_text(tweets)] if str(tweets).strip() else []

    metadata = " </s> ".join(
        [
            created_at,
            location,
            name,
            protected,
            followers_count,
            following_count,
            listed_count,
            statuses_count,
            screen_name,
            verified,
        ]
    )
    tweet_block = " </s> ".join(tweet_texts) if tweet_texts else "None"
    return f"METADATA: {metadata} </s> DESCRIPTION: {description} TWEET: {tweet_block}"


def _iter_json_array(path: Path, chunk_size: int):
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
                    raise ValueError(f"Expected JSON array in {path}")
                cursor += 1
                started = True
            progressed = False
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor >= len(buffer):
                    break
                current = buffer[cursor]
                if current == ",":
                    cursor += 1
                    progressed = True
                    continue
                if current == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    break
                yield value
                cursor = end
                progressed = True
            if eof:
                tail = buffer[cursor:].strip()
                if tail in {"", "]"}:
                    return
                raise ValueError(f"Unexpected trailing data in {path}")
            if not progressed and not buffer:
                continue


def _load_labeled_split(dataset_root: Path):
    split = _read_json(dataset_root / "split_new.json")
    train_ids = list(split.get("train") or [])
    valid_ids = list(split.get("val") or split.get("valid") or split.get("dev") or [])
    test_ids = list(split.get("test") or [])
    labeled_ids = train_ids + valid_ids + test_ids
    if not labeled_ids:
        raise ValueError("split_new.json did not contain train/val/test ids.")
    return train_ids, valid_ids, test_ids, labeled_ids


def _build_text_artifacts(dataset_root: Path, chunk_size: int):
    current_texts_path = dataset_root / "norm_user_text.json"
    labeled_texts = _read_json(current_texts_path)
    if not isinstance(labeled_texts, list):
        raise ValueError(f"Expected a list in {current_texts_path}")

    train_ids, valid_ids, test_ids, labeled_ids = _load_labeled_split(dataset_root)
    if len(labeled_texts) != len(labeled_ids):
        raise ValueError(
            f"Length mismatch: norm_user_text.json has {len(labeled_texts)} rows, "
            f"split_new.json has {len(labeled_ids)} labeled ids."
        )

    node_id_to_index = {}
    all_node_ids = []
    for index, node_id in enumerate(labeled_ids):
        canonical = _canonical_user_id(node_id)
        if not canonical:
            raise ValueError("Encountered empty labeled node id.")
        if canonical in node_id_to_index:
            raise ValueError(f"Duplicate labeled node id: {canonical}")
        node_id_to_index[canonical] = index
        all_node_ids.append(canonical)

    support_path = dataset_root / "support.json"
    output_text_path = dataset_root / "norm_user_text_new.json"
    support_count = 0
    with open(output_text_path, "w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for text in labeled_texts:
            if not first:
                handle.write(",\n")
            handle.write(json.dumps(text, ensure_ascii=False))
            first = False

        for record in _iter_json_array(support_path, chunk_size):
            raw_id = record.get("ID") or record.get("id") or record.get("user_id")
            canonical = _canonical_user_id(raw_id)
            if not canonical:
                raise ValueError("Support record missing ID.")
            if canonical in node_id_to_index:
                if node_id_to_index[canonical] < len(labeled_ids):
                    raise ValueError(f"Support node overlaps labeled node: {canonical}")
                raise ValueError(f"Duplicate support node id: {canonical}")
            node_id_to_index[canonical] = len(labeled_ids) + support_count
            all_node_ids.append(canonical)
            if not first:
                handle.write(",\n")
            handle.write(json.dumps(_support_profile_to_text(record), ensure_ascii=False))
            first = False
            support_count += 1

        handle.write("\n]\n")

    total_nodes = len(labeled_ids) + support_count
    support_idx_start = len(labeled_ids)
    support_idx_end = total_nodes - 1

    support_idx = _to_index_tensor(range(support_idx_start, total_nodes))

    return {
        "norm_user_text_new_path": str(output_text_path),
        "node_id_to_index": node_id_to_index,
        "all_node_ids": all_node_ids,
        "support_idx": support_idx,
        "support_count": support_count,
        "labeled_count": len(labeled_ids),
        "total_node_count": total_nodes,
        "support_idx_start": support_idx_start,
        "support_idx_end": support_idx_end,
        "split_counts": {
            "train": len(train_ids),
            "valid": len(valid_ids),
            "test": len(test_ids),
            "support": support_count,
        },
        "label_text_path": str(current_texts_path),
    }


def _to_index_tensor(values):
    import torch

    return torch.tensor(list(values), dtype=torch.long)


def _write_edge_json_group(handle, source_id, entries, first_group):
    if not entries:
        return first_group
    if not first_group:
        handle.write(",\n")
    handle.write(json.dumps(source_id, ensure_ascii=False))
    handle.write(": ")
    handle.write(json.dumps(entries, ensure_ascii=False))
    return False


def _build_edge_artifacts(dataset_root: Path, node_id_to_index: dict[str, int], all_node_ids, chunk_size: int):
    import torch

    edge_csv_path = _resolve_official_edge_csv_path(dataset_root)
    csv.field_size_limit(1 << 30)

    reverse_follow = {}
    src_edges = array("I")
    dst_edges = array("I")
    edge_types = array("B")
    relation_counts = {"post": 0, "friend": 0, "follow": 0, "unknown": 0}
    dropped_unknown_user_edges = 0
    kept_friend_edges = 0
    kept_follow_edges = 0

    with open(edge_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = _canonical_user_id(row["source_id"])
            relation = str(row["relation"]).strip()
            target_id = _canonical_user_id(row["target_id"])
            if relation not in relation_counts:
                relation_counts["unknown"] += 1
                continue
            relation_counts[relation] += 1
            if relation == "friend":
                src_index = node_id_to_index.get(source_id)
                dst_index = node_id_to_index.get(target_id)
                if src_index is None or dst_index is None:
                    dropped_unknown_user_edges += 1
                    continue
                src_edges.append(int(src_index))
                dst_edges.append(int(dst_index))
                edge_types.append(1)
                kept_friend_edges += 1
            elif relation == "follow":
                reverse_follow.setdefault(target_id, []).append(source_id)
                src_index = node_id_to_index.get(target_id)
                dst_index = node_id_to_index.get(source_id)
                if src_index is None or dst_index is None:
                    dropped_unknown_user_edges += 1
                    continue
                src_edges.append(int(src_index))
                dst_edges.append(int(dst_index))
                edge_types.append(0)
                kept_follow_edges += 1

    edge_json_path = dataset_root / "edge_new.json"
    seen_sources = set()
    written_groups = 0
    first_group = True
    current_source = None
    current_entries = None

    with open(edge_json_path, "w", encoding="utf-8") as handle:
        handle.write("{")
        handle.write(json.dumps("source_id", ensure_ascii=False))
        handle.write(": ")
        handle.write(json.dumps([["relation", "target_id"]], ensure_ascii=False))
        first_group = False

        with open(edge_csv_path, "r", encoding="utf-8", newline="") as edge_handle:
            reader = csv.DictReader(edge_handle)
            for row in reader:
                source_id = _canonical_user_id(row["source_id"])
                relation = str(row["relation"]).strip()
                target_id = _canonical_user_id(row["target_id"])
                if current_source is None:
                    current_source = source_id
                    current_entries = [["follow", follower_id] for follower_id in reverse_follow.pop(source_id, [])]
                elif source_id != current_source:
                    if source_id in seen_sources:
                        raise ValueError("edge.csv is not grouped by source_id; cannot stream edge_new.json safely.")
                    first_group = _write_edge_json_group(handle, current_source, current_entries, first_group)
                    if current_entries:
                        written_groups += 1
                    seen_sources.add(current_source)
                    current_source = source_id
                    current_entries = [["follow", follower_id] for follower_id in reverse_follow.pop(source_id, [])]

                if relation == "post":
                    current_entries.append(["post", target_id])
                elif relation == "friend":
                    current_entries.append(["friend", target_id])
                elif relation == "follow":
                    continue
                else:
                    current_entries.append([relation, target_id])

            if current_source is not None:
                first_group = _write_edge_json_group(handle, current_source, current_entries, first_group)
                if current_entries:
                    written_groups += 1
                seen_sources.add(current_source)

        for source_id in all_node_ids:
            follow_entries = reverse_follow.pop(source_id, None)
            if not follow_entries:
                continue
            first_group = _write_edge_json_group(
                handle,
                source_id,
                [["follow", follower_id] for follower_id in follow_entries],
                first_group,
            )
            written_groups += 1

        for source_id, follow_entries in list(reverse_follow.items()):
            if not follow_entries:
                continue
            first_group = _write_edge_json_group(
                handle,
                source_id,
                [["follow", follower_id] for follower_id in follow_entries],
                first_group,
            )
            written_groups += 1
        handle.write("}\n")

    edge_index = torch.stack(
        [
            torch.tensor(src_edges, dtype=torch.long),
            torch.tensor(dst_edges, dtype=torch.long),
        ],
        dim=0,
    )
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_manifest = {
        "edge_source_mode": "official_format22_edge_csv_normalized",
        "edge_csv_path": str(edge_csv_path),
        "edge_new_json_path": str(edge_json_path),
        "edge_json_group_count": int(written_groups),
        "kept_user_user_edges": int(edge_type.numel()),
        "kept_friend_edges": int(kept_friend_edges),
        "kept_follow_edges": int(kept_follow_edges),
        "dropped_unknown_user_edges": int(dropped_unknown_user_edges),
        "raw_edge_csv_counts": relation_counts,
        "paper_table_user_user_relation_rows": int(relation_counts["friend"] + relation_counts["follow"]),
        "paper_table_bidirected_user_user_edges_if_doubled": int(2 * (relation_counts["friend"] + relation_counts["follow"])),
        "edge_type_semantics": {
            "1": "source follows target",
            "0": "source is a follower of target",
        },
        "edge_json_normalization": {
            "friend": "kept as source_id -> target_id",
            "follow": "normalized from source_id,target_id to target_id -> source_id",
            "post": "kept as source_id -> target_id",
        },
    }
    return edge_index, edge_type, edge_manifest


def _run_artifacts(dataset_root: Path, chunk_size: int):
    artifacts = _build_text_artifacts(dataset_root, chunk_size)
    edge_index, edge_type, edge_manifest = _build_edge_artifacts(
        dataset_root=dataset_root,
        node_id_to_index=artifacts["node_id_to_index"],
        all_node_ids=artifacts["all_node_ids"],
        chunk_size=chunk_size,
    )

    _write_torch(dataset_root / "edge_index_new.pt", edge_index)
    _write_torch(dataset_root / "edge_type_new.pt", edge_type)
    _write_torch(dataset_root / "support_idx.pt", artifacts["support_idx"])

    manifest = {
        "dataset_root": str(dataset_root),
        "source_text_path": artifacts["label_text_path"],
        "norm_user_text_new_path": artifacts["norm_user_text_new_path"],
        "support_source_path": str(dataset_root / "support.json"),
        "labeled_count": artifacts["labeled_count"],
        "support_count": artifacts["support_count"],
        "total_node_count": artifacts["total_node_count"],
        "reused_labeled_text_prefix": True,
        "split_counts": artifacts["split_counts"],
        "support_idx_range": [artifacts["support_idx_start"], artifacts["support_idx_end"]],
        "support_overlap_check": "passed",
    }
    manifest.update(edge_manifest)
    _write_json(dataset_root / "preprocess_manifest_new.json", manifest)

    return artifacts, manifest


def _select_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _encode_support(
    dataset_root: Path,
    roberta_model_path: str,
    roberta_checkpoint_path: Path | None,
    batch_size: int,
    max_length: int,
    device_name: str,
    chunk_size: int,
):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _select_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_path, use_fast=True)
    model = AutoModel.from_pretrained(roberta_model_path)
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS[:3]})
    added += tokenizer.add_tokens(SPECIAL_TOKENS[3:])
    if added:
        model.resize_token_embeddings(len(tokenizer))
    checkpoint_report = None
    if roberta_checkpoint_path is not None:
        checkpoint_report = _load_roberta_backbone_checkpoint(model, Path(roberta_checkpoint_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
    model.to(device)
    model.eval()

    embeddings = []
    batch_texts = []
    support_rows = 0

    for record in _iter_json_array(dataset_root / "support.json", chunk_size):
        text = _support_profile_to_text(record)
        batch_texts.append(text)
        support_rows += 1
        if len(batch_texts) >= int(batch_size):
            embeddings.append(_encode_batch(model, tokenizer, batch_texts, max_length, device))
            batch_texts = []

    if batch_texts:
        embeddings.append(_encode_batch(model, tokenizer, batch_texts, max_length, device))

    support_embeddings = torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, model.config.hidden_size)
    output_path = dataset_root / "support_roberta_embeddings_new.pt"
    torch.save(support_embeddings, output_path)
    manifest = {
        "dataset_root": str(dataset_root),
        "roberta_model_path": roberta_model_path,
        "roberta_checkpoint_path": str(roberta_checkpoint_path) if roberta_checkpoint_path else "",
        "pooling": "mean_last_hidden",
        "token_contract": "current_llmbot_roberta",
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "device": str(device),
        "encoded_support_count": int(support_rows),
        "output_path": str(output_path),
        "special_tokens_added": SPECIAL_TOKENS,
    }
    if checkpoint_report is not None:
        manifest["checkpoint_load"] = checkpoint_report
    _write_json(dataset_root / "support_roberta_embedding_manifest_new.json", manifest)
    return support_embeddings, manifest


def _encode_batch(model, tokenizer, texts, max_length, device):
    import torch

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        max_length=int(max_length),
        truncation=True,
        padding=True,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.no_grad():
        outputs = model(**tokenized, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    embedding = hidden.mean(dim=1).detach().cpu().float()
    return embedding


def main():
    args = _parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if args.mode == "artifacts":
        _run_artifacts(dataset_root, args.chunk_size)
        print(f"Saved support-extended artifacts under {dataset_root}")
        return

    if not args.roberta_model_path:
        raise ValueError("--roberta_model_path is required when --mode encode_support.")
    _encode_support(
        dataset_root=dataset_root,
        roberta_model_path=args.roberta_model_path,
        roberta_checkpoint_path=args.roberta_checkpoint_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_name=args.device,
        chunk_size=args.chunk_size,
    )
    print(f"Saved support embeddings under {dataset_root}")


if __name__ == "__main__":
    main()
