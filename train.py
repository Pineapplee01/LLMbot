"""
train.py
Training Pipeline for RACE-Bot-D3F
  - TextFinalSemanticHead (final-layer calibrated text branch)
  - GraphEvidenceEncoder  (RGCN + LayerNorm + residual)
  - GraphReliabilityHead  (graph-native structural reliability)
  - D3FFusion             (reliability-aware routing + residual T)
"""

import json
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from dataclasses import dataclass, fields
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from utils import EdlLoss, compute_ece, compute_brier, compute_nll, compute_aurc


# ── MiniBatch ─────────────────────────────────────────────────────────────────
@dataclass
class MiniBatch:
    """Minimal sub-graph batch for RACE-Bot-D3F (final-layer only)."""
    n_id: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    batch_size: int
    x: torch.Tensor               # node features for full sub-graph
    q_final: torch.Tensor         # final-layer LLM embedding
    struct_feats: torch.Tensor    # [N, 5] pre-concatenated graph-native features

    def to(self, device):
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(device))
        return self


# ── Neighbor Sampler ──────────────────────────────────────────────────────────
class NeighborSampler(DataLoader):
    """Pure-Python neighbor sampler (no pyg-lib / torch-sparse dependency)."""

    def __init__(self, data, sizes, batch_size, input_nodes, shuffle=True):
        self.data = data
        self.edge_index = data.edge_index
        self.edge_type = getattr(data, 'edge_type', None)
        self.sizes = sizes
        self.num_nodes = int(data.x.size(0))

        # Build adjacency list (target → sources)
        self.adj = [[] for _ in range(self.num_nodes)]
        self.adj_t = [[] for _ in range(self.num_nodes)]

        rows, cols = self.edge_index[0].tolist(), self.edge_index[1].tolist()
        types = self.edge_type.tolist() if self.edge_type is not None else None

        if types is not None:
            for r, c, t in zip(rows, cols, types):
                self.adj[c].append(r)
                self.adj_t[c].append(t)
        else:
            for r, c in zip(rows, cols):
                self.adj[c].append(r)

        super().__init__(
            TensorDataset(input_nodes.cpu()),
            batch_size=batch_size, shuffle=shuffle,
            collate_fn=self.sample_subgraph,
        )

    def sample_subgraph(self, batch_indices):
        seed_nodes = [item[0].item() for item in batch_indices]
        batch_size = len(seed_nodes)

        n_id = list(seed_nodes)
        n_id_set = set(seed_nodes)
        node_map = {n: i for i, n in enumerate(n_id)}
        edges_src, edges_dst, edges_type = [], [], []

        frontier = seed_nodes
        for size in self.sizes:
            nxt = []
            for target_node in frontier:
                neighbors = self.adj[target_node]
                if not neighbors:
                    continue
                idxs = (np.random.choice(len(neighbors), size, replace=False)
                        if len(neighbors) > size else range(len(neighbors)))
                for idx in idxs:
                    source_node = neighbors[idx]
                    if source_node not in n_id_set:
                        n_id_set.add(source_node)
                        node_map[source_node] = len(n_id)
                        n_id.append(source_node)
                        nxt.append(source_node)
                    edges_src.append(node_map[source_node])
                    edges_dst.append(node_map[target_node])
                    if self.edge_type is not None:
                        edges_type.append(self.adj_t[target_node][idx])
            frontier = nxt

        n_id_t = torch.tensor(n_id, dtype=torch.long)

        edge_index = (torch.tensor([edges_src, edges_dst], dtype=torch.long)
                       if edges_src else torch.empty((2, 0), dtype=torch.long))
        edge_type = (torch.tensor(edges_type, dtype=torch.long)
                     if self.edge_type is not None and edges_type else None)

        return MiniBatch(
            n_id=n_id_t,
            edge_index=edge_index,
            edge_type=edge_type,
            batch_size=batch_size,
            x=self.data.x[n_id_t],
            q_final=self.data.q_final[n_id_t],
            struct_feats=self.data.struct_feats[n_id_t],
        )


# ── R-EDL Loss ────────────────────────────────────────────────────────────────
def r_edl_loss(alpha, target, num_classes=2):
    """R-EDL Loss: squared error + variance term."""
    y = F.one_hot(target, num_classes=num_classes).float()
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S
    err = torch.sum((y - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return torch.mean(err + var)


# ── Trainer ───────────────────────────────────────────────────────────────────
class RACEBotTrainer:
    """Training loop for RACE-Bot-D3F (reliability-aware routing + calibration)."""

    def __init__(self, data, data_dict, model, device, epochs, lr,
                 weight_decay, batch_size, cfg, ckpt_filepath='best_model.pt'):
        self.device = device
        self.epochs = epochs
        self.labels = data.y.long().to(device)
        self.cfg = cfg
        self.ckpt_filepath = Path(ckpt_filepath)

        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self.train_loader = NeighborSampler(
            data, sizes=[10, 10], batch_size=batch_size,
            input_nodes=data_dict['train_idx'], shuffle=True,
        )
        self.val_loader = NeighborSampler(
            data, sizes=[10, 10], batch_size=batch_size,
            input_nodes=data_dict['valid_idx'], shuffle=False,
        )
        self.test_loader = NeighborSampler(
            data, sizes=[10, 10], batch_size=batch_size,
            input_nodes=data_dict['test_idx'], shuffle=False,
        )

    # ── Training ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            labels = self.labels[batch.n_id[:batch.batch_size]]

            out = self.model(batch, self.cfg.get('text'))

            loss_fused = r_edl_loss(out['alpha_fused'], labels)
            loss_text  = r_edl_loss(out['alpha_text'],  labels)
            loss_graph = r_edl_loss(out['alpha_graph'], labels)

            loss = (loss_fused
                    + self.cfg['loss']['lambda_text']  * loss_text
                    + self.cfg['loss']['lambda_graph'] * loss_graph)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    # ── Evaluation (calibration-first) ────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, split='val', save_per_node=False):
        """
        Evaluate on val or test split.

        If save_per_node=True, dumps per-node diagnostics to
        <ckpt_dir>/per_node_<split>.jsonl for downstream analysis.
        """
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader

        all_preds, all_targets = [], []
        all_probs = []
        stats = {'u_fused': [], 'conflict': [], 'temperature': [], 'rho': []}
        per_node_records = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch, self.cfg.get('text'))
            y = self.labels[batch.n_id[:batch.batch_size]]
            bsz = batch.batch_size

            prob = out['alpha_fused'] / out['alpha_fused'].sum(dim=1, keepdim=True)
            preds = prob.argmax(dim=-1)

            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

            stats['u_fused'].extend(out['u_fused'].squeeze().cpu().tolist())
            stats['conflict'].extend(out['conflict'].squeeze().cpu().tolist())
            stats['temperature'].extend(out['temperature'].squeeze().cpu().tolist())
            stats['rho'].extend(out['rho'].squeeze().cpu().tolist())

            # Per-node records for analysis
            if save_per_node:
                n_ids = batch.n_id[:bsz].cpu().tolist()
                sf = batch.struct_feats[:bsz].cpu()
                for i in range(bsz):
                    per_node_records.append({
                        'node_id': n_ids[i],
                        'y': y[i].item(),
                        'pred': preds[i].item(),
                        'conf': prob[i].max().item(),
                        'u_text': out['u_text'][i].item(),
                        'r_graph': out['r_graph'][i].item(),
                        'conflict': out['conflict'][i].item(),
                        'temperature': out['temperature'][i].item(),
                        'in_log_degree': sf[i, 0].item(),
                        'out_log_degree': sf[i, 1].item(),
                        'total_log_degree': sf[i, 2].item(),
                        'sym_homophily': sf[i, 3].item(),
                        'graph_missing': sf[i, 4].item(),
                    })

        p = torch.cat(all_preds)
        t = torch.cat(all_targets)
        probs_cat = torch.cat(all_probs)

        # ── Classification metrics ──
        metrics = {
            'f1':  f1_score(t, p, average='macro'),
            'acc': accuracy_score(t, p),
        }

        # ── Calibration metrics ──
        try:
            metrics['ece']   = compute_ece(probs_cat, t)
            metrics['brier'] = compute_brier(probs_cat, t)
            metrics['nll']   = compute_nll(probs_cat, t)

            confs = probs_cat.max(dim=1).values
            correctness = (p == t).float()
            metrics['aurc'] = compute_aurc(confs, correctness)
        except Exception:
            pass

        # ── Stream statistics ──
        metrics.update({k: float(np.mean(v)) for k, v in stats.items()})

        # ── Per-node dump ──
        if save_per_node and per_node_records:
            dump_path = self.ckpt_filepath.parent / f'per_node_{split}.jsonl'
            with open(dump_path, 'w') as f:
                for rec in per_node_records:
                    f.write(json.dumps(rec) + '\n')
            print(f"  💾 Per-node outputs saved to {dump_path}")

        return metrics

    # ── Post-hoc temperature calibration ─────────────────────────────────

    @torch.no_grad()
    def _collect_logits(self, loader):
        """Collect fused logits and labels from a loader (no grad)."""
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch, self.cfg.get('text'))
            y = self.labels[batch.n_id[:batch.batch_size]]
            # Convert alpha to logits: log(alpha / S)
            alpha = out['alpha_fused']
            logits = torch.log(alpha / alpha.sum(dim=1, keepdim=True) + 1e-8)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
        return torch.cat(all_logits), torch.cat(all_labels)

    def posthoc_calibrate(self, max_iter=50, lr=0.01):
        """
        Learn a single post-hoc temperature on the validation set
        (model frozen). Returns the optimal T scalar.
        """
        logits, labels = self._collect_logits(self.val_loader)
        T_param = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T_param], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits / T_param.clamp(min=0.1)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_opt = T_param.item()
        print(f"  Post-hoc T = {T_opt:.4f}")
        return T_opt

    @torch.no_grad()
    def evaluate_calibrated(self, T_posthoc, split='test', save_per_node=False):
        """Evaluate with post-hoc temperature applied to fused alpha."""
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader

        all_preds, all_targets, all_probs = [], [], []
        per_node_records = []

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch, self.cfg.get('text'))
            y = self.labels[batch.n_id[:batch.batch_size]]
            bsz = batch.batch_size

            # Apply post-hoc T to fused alpha
            alpha = out['alpha_fused']
            log_p = torch.log(alpha / alpha.sum(dim=1, keepdim=True) + 1e-8)
            prob = F.softmax(log_p / T_posthoc, dim=1)
            preds = prob.argmax(dim=-1)

            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

            if save_per_node:
                n_ids = batch.n_id[:bsz].cpu().tolist()
                sf = batch.struct_feats[:bsz].cpu()
                for i in range(bsz):
                    per_node_records.append({
                        'node_id': n_ids[i],
                        'y': y[i].item(),
                        'pred': preds[i].item(),
                        'conf': prob[i].max().item(),
                        'u_text': out['u_text'][i].item(),
                        'r_graph': out['r_graph'][i].item(),
                        'conflict': out['conflict'][i].item(),
                        'temperature': out['temperature'][i].item(),
                        'T_posthoc': T_posthoc,
                        'in_log_degree': sf[i, 0].item(),
                        'out_log_degree': sf[i, 1].item(),
                        'total_log_degree': sf[i, 2].item(),
                        'sym_homophily': sf[i, 3].item(),
                        'graph_missing': sf[i, 4].item(),
                    })

        p = torch.cat(all_preds)
        t = torch.cat(all_targets)
        probs_cat = torch.cat(all_probs)

        metrics = {
            'f1': f1_score(t, p, average='macro'),
            'acc': accuracy_score(t, p),
        }
        try:
            metrics['ece'] = compute_ece(probs_cat, t)
            metrics['brier'] = compute_brier(probs_cat, t)
            metrics['nll'] = compute_nll(probs_cat, t)
            confs = probs_cat.max(dim=1).values
            correctness = (p == t).float()
            metrics['aurc'] = compute_aurc(confs, correctness)
        except Exception:
            pass

        if save_per_node and per_node_records:
            dump_path = self.ckpt_filepath.parent / f'per_node_{split}_calibrated.jsonl'
            with open(dump_path, 'w') as f:
                for rec in per_node_records:
                    f.write(json.dumps(rec) + '\n')

        return metrics

    # ── Composite checkpoint score ────────────────────────────────────────

    @staticmethod
    def _checkpoint_score(metrics):
        """
        Reliability-aware composite score for model selection.
        Higher is better.  F1 dominates, but ECE/AURC penalties prevent
        selecting a model that is accurate but badly calibrated.
        """
        f1 = metrics.get('f1', 0)
        ece = metrics.get('ece', 0)
        aurc = metrics.get('aurc', 0)
        return f1 - 0.05 * ece - 0.05 * aurc

    # ── Main loop ─────────────────────────────────────────────────────────

    def train(self):
        best_score = -float('inf')
        print("\n🚀 Starting RACE-Bot-D3F Training...")
        for epoch in range(1, self.epochs + 1):
            loss = self._train_epoch(epoch)
            val = self.evaluate('val')
            score = self._checkpoint_score(val)

            if wandb.run:
                wandb.log({
                    'epoch': epoch, 'train/loss': loss,
                    **{f'val/{k}': v for k, v in val.items()},
                    'val/ckpt_score': score,
                })

            ece_str = f" | ECE: {val['ece']:.4f}" if 'ece' in val else ""
            aurc_str = f" | AURC: {val['aurc']:.4f}" if 'aurc' in val else ""
            print(
                f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
                f"F1: {val['f1']:.4f} | Acc: {val['acc']:.4f}"
                f"{ece_str}{aurc_str} | "
                f"Score: {score:.4f} | "
                f"U: {val.get('u_fused', 0):.3f} | "
                f"C: {val.get('conflict', 0):.3f}"
            )

            if score > best_score:
                best_score = score
                torch.save(
                    {'model_state_dict': self.model.state_dict(),
                     'best_val_score': best_score,
                     'best_val_metrics': val},
                    self.ckpt_filepath,
                )

        # ── Reload best checkpoint before test ──
        print(f"\n Reloading best checkpoint (score={best_score:.4f})...")
        ckpt = torch.load(self.ckpt_filepath, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])

        # ── Post-hoc calibration on validation set ──
        print("  Learning post-hoc temperature on validation set...")
        T_posthoc = self.posthoc_calibrate()

        # ── Final test (uncalibrated) with per-node dump ──
        test_raw = self.evaluate('test', save_per_node=True)
        print(f"\n Test Results (uncalibrated):")
        print(f"   F1={test_raw['f1']:.4f} | Acc={test_raw['acc']:.4f}"
              + (f" | ECE={test_raw['ece']:.4f}" if 'ece' in test_raw else "")
              + (f" | Brier={test_raw['brier']:.4f}" if 'brier' in test_raw else "")
              + (f" | NLL={test_raw['nll']:.4f}" if 'nll' in test_raw else "")
              + (f" | AURC={test_raw['aurc']:.4f}" if 'aurc' in test_raw else ""))

        # ── Final test (post-hoc calibrated) ──
        test_cal = self.evaluate_calibrated(T_posthoc, 'test', save_per_node=True)
        print(f"\n Test Results (post-hoc T={T_posthoc:.4f}):")
        print(f"   F1={test_cal['f1']:.4f} | Acc={test_cal['acc']:.4f}"
              + (f" | ECE={test_cal['ece']:.4f}" if 'ece' in test_cal else "")
              + (f" | Brier={test_cal['brier']:.4f}" if 'brier' in test_cal else "")
              + (f" | NLL={test_cal['nll']:.4f}" if 'nll' in test_cal else "")
              + (f" | AURC={test_cal['aurc']:.4f}" if 'aurc' in test_cal else ""))

        if wandb.run:
            wandb.log({f'test_raw/{k}': v for k, v in test_raw.items()})
            wandb.log({f'test_cal/{k}': v for k, v in test_cal.items()})
            wandb.log({'posthoc_T': T_posthoc})

        return best_score