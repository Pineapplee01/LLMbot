import argparse
import json
import os
from pathlib import Path

import torch

from model_building import build_LM_model
from runtime_env import now_iso, write_json_file
from utils import load_raw_data


def tokenize(tokenizer, texts, max_length, device):
    return tokenizer(
        texts,
        return_tensors="pt",
        max_length=int(max_length),
        padding=True,
        truncation=True,
    ).to(device)


def write_json(path, payload):
    write_json_file(path, payload)


def extract(args):
    device = torch.device(f"cuda:{args.device}" if int(args.device) >= 0 and torch.cuda.is_available() else "cpu")
    data = load_raw_data(args.dataset, use_GNN=True, graph_data_variant=args.graph_data_variant)
    texts = list(data["user_text"])
    labels = data["labels"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "lm_model": "roberta",
        "classifier_n_layers": 2,
        "classifier_hidden_dim": 128,
        "activation": "relu",
        "lm_dropout": 0.0,
        "att_dropout": 0.0,
        "pretrained_model_source": args.model_path or None,
        "device": device,
    }
    model, tokenizer = build_LM_model(model_config)
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), int(args.batch_size)):
            batch_texts = texts[start : start + int(args.batch_size)]
            tokenized = tokenize(tokenizer, batch_texts, args.max_length, device)
            embedding, _ = model(tokenized)
            rows.append(embedding.detach().cpu())
    embeddings = torch.cat(rows, dim=0).contiguous()
    if int(embeddings.shape[0]) != int(labels.shape[0]):
        raise ValueError(f"Embedding rows {embeddings.shape[0]} do not match labels {labels.shape[0]}.")

    embedding_path = output_dir / "raw_roberta_embeddings.pt"
    torch.save(embeddings, embedding_path)
    manifest = {
        "created_at": now_iso(),
        "dataset": args.dataset,
        "graph_data_variant": args.graph_data_variant,
        "model": "roberta-base",
        "model_path": args.model_path,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "embedding_shape": list(embeddings.shape),
        "embedding_path": str(embedding_path),
        "contract": "raw_roberta_embedding_extract_v1",
        "pooling": "last_hidden_state_mean_over_padded_sequence_matches_LM_Model.forward",
        "notes": "Extract-only raw RoBERTa backbone for w/o LM Supervised Fine-tuning ablation; no supervised LM update is run.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="TwiBot-20")
    parser.add_argument("--graph_data_variant", default="labeled")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default=os.environ.get("ROBERTA_BASE_PATH", ""))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
