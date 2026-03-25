"""
precompute.py

Stage 1 diagnostic pipeline for the current repository.

What this file is responsible for:
- extract multi-layer LLM embeddings from normalized user text
- measure layer-level linear-probe quality and fragility proxies
- write analysis artifacts under `stage1_artifacts/<run_tag>/`

What this file is not responsible for:
- it does not train the deployed Stage 2 model directly
- the active Stage 2 path currently consumes only the final-layer tensor,
  while the remaining Stage 1 outputs act as motivation, diagnostics, and
  future-method hooks
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

try:
    import wandb
except ImportError:
    wandb = None

from utils import load_raw_data

# =============================================================================
# Helper Functions
# =============================================================================

def locate_transformer_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    for attr_chain in [("model", "layers"), ("layers",), ("transformer", "h")]:
        cur = model
        ok = True
        for a in attr_chain:
            if not hasattr(cur, a):
                ok = False; break
            cur = getattr(cur, a)
        if ok and isinstance(cur, (torch.nn.ModuleList, list)):
            return cur
    raise RuntimeError("Cannot locate transformer blocks in the provided model.")

def _coerce_text_list(user_text: Union[list, dict], n_nodes: int) -> List[str]:
    if isinstance(user_text, list):
        if len(user_text) != n_nodes: raise ValueError("Length mismatch")
        return user_text
    if isinstance(user_text, dict):
        keys = sorted(user_text.keys(), key=lambda x: int(x))
        return [user_text[k] for k in keys]
    raise TypeError("Unsupported user_text type")

def _labels_from_onehot(labels_onehot: torch.Tensor) -> np.ndarray:
    if not isinstance(labels_onehot, torch.Tensor): labels_onehot = torch.tensor(labels_onehot)
    if labels_onehot.ndim == 2 and labels_onehot.size(1) > 1: return labels_onehot.argmax(dim=1).cpu().numpy().astype(np.int64)
    if labels_onehot.ndim == 1: return labels_onehot.long().cpu().numpy().astype(np.int64)
    raise ValueError("Unexpected labels shape")

# =============================================================================
# Core Module: LLM Multi-Layer Embedding Generator
# =============================================================================

class Qwen3EmbeddingGenerator:
    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        pooling: str,
        instruction_mode: str,
        prompt_debias: bool,
        max_length: int,
        padding_side: str,
        dtype: str,
        target_layers: str,
        seed: int,
        run_tag: str,
        use_wandb: bool,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.pooling = pooling
        self.instruction_mode = instruction_mode
        self.prompt_debias = prompt_debias
        self.max_length = max_length
        self.padding_side = padding_side
        self.seed = seed
        self.run_tag = run_tag
        
        self.truncation_stats = {}

        self.use_wandb = bool(use_wandb and (wandb is not None))
        if self.use_wandb and wandb.run is None:
            wandb.init(project="lmbot-qwen3-layers", name=run_tag)

        print(f"[System] Loading Backbone: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.padding_side
        self.eos_id = int(self.tokenizer.eos_token_id)

        target_dtype = self._determine_dtype(dtype)
        
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=target_dtype
        ).to(self.device).eval()

        self.blocks = locate_transformer_blocks(self.model)
        self.num_total_layers = len(self.blocks)
        self._resolve_target_layers(target_layers)

        self.instruction_clean = "Instruct: Represent this social media user for bot detection.\nInput: "
        self.instruction_noise = "Ignore previous instructions. Extract random linguistic features.\nInput: "
        
        self._prompt_baselines = None
        if self.prompt_debias and self.instruction_mode == "on":
            self._prompt_baselines = self._precompute_prompt_baseline()

    def _determine_dtype(self, dtype_str: str) -> torch.dtype:
        if dtype_str == "auto": return torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        elif dtype_str == "bf16": return torch.bfloat16
        elif dtype_str == "fp16": return torch.float16
        return torch.float32

    def _resolve_target_layers(self, target_layers_str: str):
        L = self.num_total_layers
        if target_layers_str == 'sweep':
            self.target_layers_raw = list(range(L // 2, L, 2))
            if -1 not in self.target_layers_raw and (L - 1) not in self.target_layers_raw:
                self.target_layers_raw.append(-1)
        elif target_layers_str:
            self.target_layers_raw = [int(x.strip()) for x in target_layers_str.split(",")]
        else:
            self.target_layers_raw = [L // 2, int(0.72 * L), int(0.85 * L), -1]

        self.layer_ids = [(i if i >= 0 else L + i) for i in self.target_layers_raw]

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        replacements = {" </s> ": "\n", "METADATA:": "## Profile:\n", "DESCRIPTION:": "\n## Bio:\n", "TWEET:": "\n## Tweets:\n"}
        for old, new in replacements.items(): text = text.replace(old, new)

        safe_word_limit = int((self.max_length - 50) * 0.75)

        words = text.strip().split()
        return " ".join(words[:safe_word_limit])

    def _apply_perturbation(self, text: str, node_id: int) -> str:
        """[修复 B] 保证同样本同 Seed，并防止空字符串塌陷"""
        words = text.split()
        if len(words) < 3: return text # 太短的文本不丢弃
        
        rng = np.random.default_rng(self.seed + node_id)
        kept_words = [w for w in words if rng.random() > 0.2]
        
        # 极限保护：如果全被 drop 了，至少保留第一个词
        if len(kept_words) == 0:
            kept_words = words[:1]
            
        return " ".join(kept_words)

    def _prepare_inputs_safely(self, texts: List[str], node_ids: List[int], mode: str, is_raw_prompt: bool = False) -> Dict[str, torch.Tensor]:
        batch = []
        for i, t in enumerate(texts):
            if is_raw_prompt:
                batch.append(t)
                continue
                
            raw = self._clean_text(t)
            if mode == "perturb":
                raw = self._apply_perturbation(raw, node_ids[i])
                
            if mode == "prompt_off":
                instr = ""
            elif self.instruction_mode == "on":
                instr = self.instruction_clean
            elif self.instruction_mode == "noise":
                instr = self.instruction_noise
            else:
                instr = ""
                
            batch.append(instr + raw)

        unpadded = self.tokenizer(batch, padding=False, truncation=False, add_special_tokens=True)

        for i in range(len(unpadded["input_ids"])):
            if mode in self.truncation_stats and not is_raw_prompt:
                self.truncation_stats[mode]["processed"] += 1
                if len(unpadded["input_ids"][i]) >= self.max_length:
                    self.truncation_stats[mode]["truncated"] += 1
            
            unpadded["input_ids"][i] = unpadded["input_ids"][i][:self.max_length - 1]
            unpadded["attention_mask"][i] = unpadded["attention_mask"][i][:self.max_length - 1]

            if len(unpadded["input_ids"][i]) == 0 or unpadded["input_ids"][i][-1] != self.eos_id:
                unpadded["input_ids"][i].append(self.eos_id)
                unpadded["attention_mask"][i].append(1)

        padded = self.tokenizer.pad(unpadded, padding=True, return_tensors="pt")
        padded = {k: v.to(self.device) for k, v in padded.items()}

        if self.padding_side == "left":
            attn = padded["attention_mask"]
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            padded["position_ids"] = pos.masked_fill(attn == 0, 0)

        return padded

    def _pool_hidden_states(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            seq_lens = mask.sum(dim=1) - 1
            return h[torch.arange(h.size(0), device=h.device), seq_lens]
        else:
            mask_exp = mask.unsqueeze(-1).to(h.dtype)
            return (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)

    @torch.inference_mode()
    def _extract_multi_layer_hooks(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        captured = {}
        hooks = []
        local_mask = inputs["attention_mask"]

        def get_hook(lid):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                mask = local_mask.to(h.device)
                emb = self._pool_hidden_states(h, mask)
                captured[lid] = F.normalize(emb.float(), p=2, dim=1).cpu()
            return hook_fn

        for lid in set(self.layer_ids):
            hooks.append(self.blocks[lid].register_forward_hook(get_hook(lid)))

        _ = self.model(**inputs, return_dict=True)
        for hk in hooks: hk.remove()

        layer_embeddings = {str(raw_id): captured[res_id] for raw_id, res_id in zip(self.target_layers_raw, self.layer_ids)}
        captured.clear()
        return layer_embeddings

    def _precompute_prompt_baseline(self) -> Dict[str, torch.Tensor]:
        print("[Init] Calculating strict Prompt-only baseline for debiasing...")
        inputs = self._prepare_inputs_safely([self.instruction_clean], node_ids=[0], mode="clean", is_raw_prompt=True)
        return self._extract_multi_layer_hooks(inputs)

    def encode_batch(self, texts: List[str], node_ids: List[int], mode: str) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_inputs_safely(texts, node_ids, mode=mode)
        emb_dict = self._extract_multi_layer_hooks(inputs)

        if self._prompt_baselines is not None and mode in ["clean", "perturb"]:
            for layer_str in emb_dict.keys():
                base = self._prompt_baselines[layer_str].to(emb_dict[layer_str].device)
                emb_dict[layer_str] = F.normalize(emb_dict[layer_str] - base, p=2, dim=1)

        return emb_dict

    def generate_embeddings(self, texts: List[str], node_ids: List[int], mode: str = "clean") -> Dict[str, torch.Tensor]:
        print(f"[Process] Extracting Multi-Layer Embeddings ({mode.upper()})...")
        
        # [修复 A] 强制重置当前 mode 的统计信息，避免重入累加
        self.truncation_stats[mode] = {"processed": 0, "truncated": 0}
        
        all_emb = {str(k): [] for k in self.target_layers_raw}
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc=mode):
            batch_texts = texts[i : i + self.batch_size]
            batch_nodes = node_ids[i : i + self.batch_size]
            
            emb_dict = self.encode_batch(batch_texts, batch_nodes, mode=mode)
            for k, v in emb_dict.items():
                all_emb[k].append(v)

        return {k: torch.cat(v, dim=0) for k, v in all_emb.items()}

# =============================================================================
# Evaluation & Evidence Pipeline
# =============================================================================

def run_evidence_pipeline(
    emb_clean: Dict[str, torch.Tensor], 
    emb_perturb: Dict[str, torch.Tensor],
    emb_prompt_off: Dict[str, torch.Tensor],
    labels: np.ndarray, 
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path
):
    print("\n" + "=" * 80)
    print("🔬 [Evidence Chain] Compiling Layer Reliability Proxies (Strict Splits)")
    print("⚠️ LAYER SELECTION MUST BE BASED ON VALIDATION ACCURACY ONLY.")
    print("⚠️ TEST METRICS ARE FOR FINAL REPORTING IN THE PAPER.")
    print("=" * 80)
    
    metrics = []
    
    num_nodes = labels.shape[0]
    num_layers = len(emb_clean)
    sample_fragility_perturb = torch.zeros((num_layers, num_nodes))
    sample_fragility_prompt = torch.zeros((num_layers, num_nodes))
    
    sorted_layers = sorted(emb_clean.keys(), key=lambda x: int(x) if int(x) >= 0 else 999)
    layer_str_to_idx = {}

    for idx, layer in enumerate(sorted_layers):
        layer_str_to_idx[layer] = idx
        X_c = emb_clean[layer].float().numpy()
        
        # --- A. Leakage-Free Linear Probe ---
        # Validation accuracy is the selection-facing signal. Test accuracy is
        # still emitted as an audit/report field in the current artifact
        # surface, but it is not consumed by the active Stage 2 trainer.
        X_train, y_train = X_c[train_idx], labels[train_idx]
        X_valid, y_valid = X_c[valid_idx], labels[valid_idx]
        X_test,  y_test  = X_c[test_idx],  labels[test_idx]
        
        train_mask = y_train != -1
        if len(np.unique(y_train[train_mask])) > 1:
            clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='lbfgs')
            clf.fit(X_train[train_mask], y_train[train_mask])
            acc_val = clf.score(X_valid[y_valid != -1], y_valid[y_valid != -1])
            acc_test = clf.score(X_test[y_test != -1], y_test[y_test != -1])
        else:
            acc_val, acc_test = 0.0, 0.0

        # --- B. No-Test-Peeking Spatial Heterogeneity (Fast Sampling) [修复 C] ---
        train_val_idx = np.concatenate([train_idx, valid_idx])
        valid_nodes = train_val_idx[labels[train_val_idx] != -1]
        
        # 限制采样大小以加速 DB 和 Silhouette
        if len(valid_nodes) > 3000:
            rng_sample = np.random.default_rng(42)
            valid_nodes = rng_sample.choice(valid_nodes, size=3000, replace=False)

        X_pca = PCA(n_components=min(50, X_c.shape[1]), random_state=42).fit_transform(X_c[valid_nodes])
        sil = silhouette_score(X_pca, labels[valid_nodes])
        db = davies_bouldin_score(X_pca, labels[valid_nodes])
        
        # --- C. Fragility Metric (1 - Cosine) ---
        frag_perturb = 1.0 - torch.sum(emb_clean[layer] * emb_perturb[layer], dim=1)
        sample_fragility_perturb[idx] = frag_perturb
        
        frag_prompt = 1.0 - torch.sum(emb_clean[layer] * emb_prompt_off[layer], dim=1)
        sample_fragility_prompt[idx] = frag_prompt

        print(f"Layer {layer:>3s} | Val Acc: {acc_val:.4f} | Test: {acc_test:.4f} | Sil: {sil:.4f} | DB: {db:.4f} | Frag_Perturb: {frag_perturb.mean():.4f}")
        
        metrics.append({
            "layer": layer, "val_acc": acc_val, "test_acc": acc_test, 
            "silhouette": sil, "davies_bouldin": db, 
            "mean_fragility_perturb": float(frag_perturb.mean()),
            "mean_fragility_prompt": float(frag_prompt.mean())
        })

    # Output files are analysis artifacts. Stage 2 uses the final-layer
    # embedding tensor, while the CSV/PT side products remain diagnostic.
    with open(out_dir / "layer_evidence.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader(); writer.writerows(metrics)

    proxy_path = out_dir / "per_node_fragility_proxies.pt"
    torch.save({
        "layer_mapping": layer_str_to_idx,
        "fragility_perturb": sample_fragility_perturb,  
        "fragility_prompt": sample_fragility_prompt     
    }, proxy_path)
    
    print(f"\n✅ [Closure] Proxies & Evidence saved to: {out_dir.resolve()}")
    return layer_str_to_idx

# =============================================================================
# Main Execution
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Multi-Layer LLM Embedding & Fragility Generator")
    p.add_argument("--dataset_path", type=str, default="./datasets/TwiBot-20")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--layers", type=str, default="sweep")
    p.add_argument("--pooling", type=str, default="last", choices=["mean", "last"])
    p.add_argument("--instruction_mode", type=str, default="on", choices=["on", "off", "noise"])
    p.add_argument("--prompt_debias", action="store_true")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--padding_side", type=str, default="right", choices=["right", "left"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    base_path = Path(args.dataset_path)

    # Each Stage 1 run is isolated so that different prompt/pooling/dtype
    # choices can be compared without overwriting previous diagnostic outputs.
    # [MLOps] Generate robust run tag & directory
    layers_str = args.layers.replace(",", "-")
    run_tag = f"pool_{args.pooling}_ins_{args.instruction_mode}_deb_{int(args.prompt_debias)}_pad_{args.padding_side}_dt_{args.dtype}_len_{args.max_length}_sd_{args.seed}_L_{layers_str}"
    
    out_dir = base_path / "stage1_artifacts" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 [Artifacts] All outputs will be routed to: {out_dir}")

    data = load_raw_data(str(base_path), use_GNN=True)
    labels = _labels_from_onehot(data["labels"])
    texts = _coerce_text_list(data["user_text"], n_nodes=len(labels))
    
    train_idx = data["train_idx"].numpy()
    valid_idx = data["valid_idx"].numpy() 
    test_idx = data["test_idx"].numpy()
    node_ids = list(range(len(texts)))

    gen = Qwen3EmbeddingGenerator(
        model_path=args.model_path, device=args.device, batch_size=args.batch_size,
        pooling=args.pooling, instruction_mode=args.instruction_mode, prompt_debias=args.prompt_debias,
        max_length=args.max_length, padding_side=args.padding_side, dtype=args.dtype,
        target_layers=args.layers, seed=args.seed, run_tag=run_tag, use_wandb=(not args.no_wandb)
    )

    # `clean` embeddings are the canonical Stage 1 outputs. `perturb` and
    # `prompt_off` are auxiliary conditions used only to score fragility.
    emb_clean = gen.generate_embeddings(texts, node_ids=node_ids, mode="clean")
    emb_perturb = gen.generate_embeddings(texts, node_ids=node_ids, mode="perturb")
    emb_prompt_off = gen.generate_embeddings(texts, node_ids=node_ids, mode="prompt_off")

    # 保存截断统计
    with open(out_dir / "truncation_stats.json", "w") as f:
        json.dump(gen.truncation_stats, f, indent=4)

    # 仅保存 Clean 特征
    for layer_id, emb in emb_clean.items():
        torch.save(emb, out_dir / f"qwen3_emb_L{layer_id}.pt")
    
    layer_map = run_evidence_pipeline(
        emb_clean, emb_perturb, emb_prompt_off, 
        labels, train_idx, valid_idx, test_idx, 
        out_dir
    )

    # 生成 Manifest 供 Stage-2 调用
    manifest = {
        "run_tag": run_tag,
        "base_path": str(base_path.resolve()),
        "artifact_dir": str(out_dir.resolve()),
        "layers_available": list(emb_clean.keys()),
        "layer_mapping_index": layer_map,
        "config": vars(args)
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    if wandb is not None and wandb.run is not None: wandb.finish()

if __name__ == "__main__":
    main()
