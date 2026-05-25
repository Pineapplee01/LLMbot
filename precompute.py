"""Precompute Qwen prompt embeddings for GLANCE-style and relation-aware views.

This helper builds per-node prompt texts, encodes them with Qwen3-Embedding,
applies last-token pooling plus optional l2 normalization, and stores a prompt
cache under a stable `embeddings` key for downstream graph_detector_prepare and
strict GLANCE runs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from utils import ensure_dir, read_json, resolve_dataset_path, tensor_sha256, write_json, write_torch


DEFAULT_QWEN_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/"
    "snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
)
DEFAULT_OUTPUT_NAME = "glance_qwen3_prompt_cache.pt"
PROMPT_MODE_CHOICES = (
    "glance_ego",
    "glance_hop1",
    "glance_hop2",
    "glance_concat_ego_hop1_hop2",
    "relation_aware_ego",
    "relation_aware_1hop",
)
NEIGHBOR_SAMPLING_CHOICES = ("auto", "uniform_random", "directional_heuristic")


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


def _default_output_path(dataset_path: Path, prompt_mode: str):
    if prompt_mode == "glance_concat_ego_hop1_hop2":
        return dataset_path / DEFAULT_OUTPUT_NAME
    return dataset_path / f"glance_qwen3_prompt_cache_{prompt_mode}.pt"


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute GLANCE-style or relation-aware Qwen prompt embeddings.")
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
    return parser


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
    return torch.cat(chunks, dim=0)


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

    device = _device_from_args(str(args.device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=bool(args.trust_remote_code))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    encoded = {}
    for component_name, component_prompts in prompt_bundle["prompt_components"].items():
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

    selected_components = list(prompt_bundle["selected_components"])
    if len(selected_components) == 1:
        selected_embeddings = encoded[selected_components[0]]
        embedding_mode = "single_component"
    else:
        selected_embeddings = torch.cat([encoded[name] for name in selected_components], dim=1)
        embedding_mode = "selected_component_concat"

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    payload = {"embeddings": selected_embeddings.to(save_dtype).contiguous()}
    for component_name, component_tensor in encoded.items():
        payload[component_name] = component_tensor.to(save_dtype).contiguous()
    write_torch(output_path, payload)

    tensor_hashes = {"embeddings": tensor_sha256(payload["embeddings"])}
    for component_name in encoded:
        tensor_hashes[component_name] = tensor_sha256(payload[component_name])

    component_budget = {}
    for component_name, group in prompt_bundle["component_max_length_group"].items():
        component_budget[component_name] = int(args.max_length_ego) if group == "ego" else int(args.max_length_hop)

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
        "edge_type_semantics": {
            "relation_1": "following source->target",
            "relation_0": "follower source->target interpreted as target receives source as follower",
            "source": "aligned to LLMbot.trainer_legacy_impl directional semantic-view helper",
        },
        "strict_stage_compatibility": {
            "payload_embeddings_key": "embeddings",
            "payload_component_keys": list(encoded.keys()),
            "same_root_requirement": "graph_detector_prepare must be rerun with this exact output_path before joint_router_refinement",
        },
        "note": (
            "glance_* modes preserve GLANCE-style prompt families; relation_aware_* modes use "
            "direction-aware social-context prompts with deterministic heuristic neighbor selection."
        ),
    }
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
