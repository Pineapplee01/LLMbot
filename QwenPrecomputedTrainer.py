import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pandas as pd
import numpy as np

from utils import EdlLoss
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.utils import degree

class QwenPrecomputedTrainer:
    def __init__(
        self,
        precomputed_embeddings, 
        gnn_model,
        fusion_model,
        data_dict,
        device,
        epochs,
        lr,
        weight_decay,
        ckpt_filepath='best_model.pt',
        lambda_avuc=0.1,
        lambda_struct=0.1,
        lambda_supcon=0.1,
        supcon_temp=0.07,
        pretrain_gnn=False,
        pretrain_llm=False,
        pretrain_gate=False,
        sample='random',
        metadata=None,
        **kwargs,
    ):
        self.device = device
        self.ckpt_filepath = Path(ckpt_filepath)
        
        # Models
        self.embeddings = precomputed_embeddings.to(device)
        self.gnn = gnn_model.to(device)
        self.fusion = fusion_model.to(device)
        
        # Configs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_avuc = lambda_avuc
        self.lambda_struct = lambda_struct
        self.lambda_supcon = lambda_supcon
        self.supcon_temp = supcon_temp
        self.use_wandb = wandb.run is not None
        
        # Flags
        self.pretrain_gnn = pretrain_gnn
        self.pretrain_llm = pretrain_llm
        self.pretrain_gate = pretrain_gate 
        
        # Data & Labels
        self.data_dict = data_dict
        self.labels = data_dict['labels'].to(device)
        if self.labels.dim() > 1 and self.labels.shape[1] > 1:
            self.labels = self.labels.argmax(dim=1)

        self.labels = self.labels.long()

        # Graph Structure
        self.edge_index = data_dict['edge_index'].to(device)
        self.edge_type = data_dict['edge_type'].to(device) if 'edge_type' in data_dict else None
        
        # Meta Features
        self.homophily = self._precompute_homophily()
        self.node_degrees = self._precompute_degrees()
        
        # DataLoaders
        self.train_idx = data_dict['train_idx']
        self.val_idx = data_dict['valid_idx']
        self.test_idx = data_dict['test_idx']
        self.dataloader = self._create_dataloader(self.train_idx, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_idx, shuffle=False)
        self.test_loader = self._create_dataloader(self.test_idx, shuffle=False)

        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.edl_loss = EdlLoss()


        self._setup_optimizer()

    # =========================================================================
    #  🛠️ Helpers & Setup
    # =========================================================================

    def _setup_optimizer(self):
        """Initial optimizer setup"""
        params = list(filter(lambda p: p.requires_grad, self.gnn.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.fusion.parameters()))
        
        if len(params) > 0:
            self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _create_dataloader(self, indices, shuffle):
        dataset = torch.utils.data.TensorDataset(indices, self.labels[indices])
        return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=shuffle)

    def _precompute_homophily(self):
        # Placeholder: Return zeros if not using homophily calculation
        # You can integrate your 'calculate_homophily' logic here if needed
        return torch.zeros(self.labels.size(0), 1).to(self.device)

    def _precompute_degrees(self):
        
        row, col = self.edge_index
        deg = degree(col, self.labels.size(0), dtype=torch.float)
        # Normalize
        deg = (deg - deg.mean()) / (deg.std() + 1e-6)
        return deg.unsqueeze(1).to(self.device)

    def freeze_module(self, module, freeze=True):
        for param in module.parameters():
            param.requires_grad = not freeze


    def train_epoch(self, epoch):
        if self.pretrain_gnn:
            return self._train_epoch_gnn(epoch)
        else:
            return self._train_epoch_fusion(epoch)

    def _train_epoch_gnn(self, epoch):
        self.gnn.train()
        total_loss = 0
        steps = 0

        for batch_nodes, batch_labels in self.dataloader:
            batch_nodes = batch_nodes.to(self.device)
            batch_labels = batch_labels.to(self.device)
            self.optimizer.zero_grad()
            
            h_gnn = self._forward_gnn(batch_nodes)
            logits = self.gnn.classifier(h_gnn)
            
            loss = self.criterion_cls(logits, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            if self.use_wandb and steps % 10 == 0:
                wandb.log({"loss/gnn_cls": loss.item()})
                
        return total_loss / steps

    def _train_epoch_fusion(self, epoch):
        self.fusion.train()
        self.gnn.train()
        
        # Strategy
        margin_weight = 1.0
        u_low, u_high = 0.30, 0.65
        
        # KL Warmup
        if epoch < 5:
            kl_weight = 0.0
        else:
            progress = min(1.0, (epoch - 5) / 5)
            kl_weight = self.lambda_struct * progress
        
        total_loss = 0
        steps = 0
        monitor = {'u_corr': [], 'u_wrong': [], 'gate': []}
        
        for batch_nodes, batch_labels in self.dataloader:
            batch_nodes = batch_nodes.to(self.device)
            batch_labels = batch_labels.to(self.device)
            self.optimizer.zero_grad()
            
            h_gnn_batch = self._forward_gnn(batch_nodes)
            h_lm_batch = self.embeddings[batch_nodes]
            
            out = self.fusion(
                lm_emb=h_lm_batch,
                gnn_emb=h_gnn_batch,
                homophily=self.homophily[batch_nodes],
                degree=self.node_degrees[batch_nodes]
            )
            
            self.last_gate_mean = out['gate_value'].mean().item()
            
            loss, loss_dict = self._compute_fusion_losses(
                out, batch_labels, h_gnn_batch, 
                kl_weight, margin_weight, u_low, u_high, epoch
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            self._update_monitor(monitor, out, batch_labels)
            if self.use_wandb and steps % 10 == 0:
                self._log_wandb(loss, loss_dict, monitor)

        # cache epoch diagnostics for outer printer
        u_c = np.mean(monitor['u_corr']) if monitor['u_corr'] else 0
        u_w = np.mean(monitor['u_wrong']) if monitor['u_wrong'] else 0
        gate = np.mean(monitor['gate']) if monitor['gate'] else 0
        self.last_epoch_gate = gate
        self.last_epoch_u_gap = u_w - u_c
        return total_loss / steps

    def _forward_gnn(self, batch_nodes):
        if self.edge_type is not None:
            h_all = self.gnn(self.embeddings, self.edge_index, self.edge_type)
        else:
            h_all = self.gnn(self.embeddings, self.edge_index)
        return h_all[batch_nodes]

    def _compute_fusion_losses(self, out, labels, h_gnn, kl_weight, margin_weight, u_low, u_high, epoch):
        loss_cls = self.criterion_cls(out['logits'], labels)


        # Auxiliary Loss
        loss_text_aux = self.criterion_cls(out['logits_lm'], labels)
        
        if hasattr(self.fusion, 'gnn_classifier'):
             z_gnn = self.fusion.gnn_proj(h_gnn) # Sequential has activation
             logits_gnn = self.fusion.gnn_classifier(z_gnn)
        else:
             logits_gnn = self.gnn.classifier(h_gnn)

        loss_graph_aux = self.criterion_cls(logits_gnn, labels)
        
        # EDL Loss
        if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
        loss_edl_text = self.edl_loss(out['alpha_text'], labels)
        
        if 'alpha_gnn' in out:
            loss_edl_graph = self.edl_loss(out['alpha_gnn'], labels)
        else:
            loss_edl_graph = torch.tensor(0.0, device=self.device)

        # VIB KL Loss
        loss_vib = out['kl_loss'] * kl_weight
        
        # AvUC Loss 
        u_text = out['u_text']
        u_graph = out['u_graph'] 
        gate = out['gate_value']
        u_fused = gate * u_text + (1 - gate) * u_graph
        
        loss_avuc = self.calculate_avuc_loss(out['logits'], labels, u_fused) * self.lambda_avuc
        
        # Gate Consistency Loss
        loss_gate = self.fusion.get_structure_consistency_loss(
            out['gate_value'], self.homophily[labels.device.index], out['u_text']
        )
        
        # Margin Loss
        loss_margin = self.calculate_uncertainty_margin_loss(
            out['logits_lm'], labels, out['u_text'], u_low, u_high
        ) * margin_weight
        
        total_loss = loss_cls + \
                     (0.5 * loss_text_aux) + \
                     (0.5 * loss_graph_aux) + \
                     (0.2 * loss_edl_text) + \
                     (0.2 * loss_edl_graph) + \
                     loss_vib + \
                     loss_avuc + \
                     (0.1 * loss_gate) + \
                     loss_margin
        
        return total_loss, {
            "cls": loss_cls, 
            "margin": loss_margin, 
            "avuc": loss_avuc,
            "edl_g": loss_edl_graph,
            "aux_txt": loss_text_aux
        }

    # =========================================================================
    #  🧪 Evaluation & Diagnosis (FIXED)
    # =========================================================================

    def evaluate(self, split='val'):
        self.gnn.eval()
        self.fusion.eval()
        
        if split == 'val': loader = self.val_loader
        elif split == 'test': loader = self.test_loader
        else: loader = self.dataloader
        
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_nodes, batch_labels = batch
                else:
                    batch_nodes = batch
                    batch_labels = self.labels[batch_nodes]

                batch_nodes = batch_nodes.to(self.device)
                
                if self.pretrain_gnn:
                    h_gnn = self._forward_gnn(batch_nodes)
                    logits = self.gnn.classifier(h_gnn)
                else:
                    h_gnn = self._forward_gnn(batch_nodes)
                    out = self.fusion(
                        lm_emb=self.embeddings[batch_nodes],
                        gnn_emb=h_gnn,
                        homophily=self.homophily[batch_nodes],
                        degree=self.node_degrees[batch_nodes]
                    )
                    logits = out['logits']
                
                preds.append(logits.argmax(dim=1).cpu())
                targets.append(batch_labels.cpu())
                
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        return f1_score(targets, preds, average='macro'), accuracy_score(targets, preds)

    def diagnose_errors(self, split='val'):
        """
        [Enhanced] Diagnosis with 4-Category Breakdown
        """
        self.gnn.eval()
        self.fusion.eval()
        
        if split == 'val': loader = self.val_loader
        elif split == 'test': loader = self.test_loader
        else: loader = self.dataloader
        
        results = []
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_nodes, batch_labels = batch
                else:
                    batch_nodes = batch
                    batch_labels = self.labels[batch_nodes]
                    
                batch_nodes = batch_nodes.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                h_lm = self.embeddings[batch_nodes]
                h_gnn = self._forward_gnn(batch_nodes)
                
                out = self.fusion(
                    lm_emb=h_lm, 
                    gnn_emb=h_gnn,
                    homophily=self.homophily[batch_nodes],
                    degree=self.node_degrees[batch_nodes]
                )
                
                # --- Prediction Logic ---
                # 1. Full Model
                probs_full = F.softmax(out['logits'], dim=1)
                conf_full, pred_full = probs_full.max(dim=1)
                
                # 2. Text Branch
                probs_text = F.softmax(out['logits_lm'], dim=1)
                conf_text, pred_text = probs_text.max(dim=1)

                probs_graph = F.softmax(out['logits_gnn'], dim=1)
                conf_graph, pred_graph = probs_graph.max(dim=1)
                
                for i in range(len(batch_nodes)):
                    # Determine Category
                    is_text_correct = (pred_text[i] == batch_labels[i]).item()
                    is_graph_correct = (pred_graph[i] == batch_labels[i]).item()
                    
                    if is_text_correct and is_graph_correct: case_type = 'Easy'
                    elif not is_text_correct and not is_graph_correct: case_type = 'Hard'
                    elif is_text_correct and not is_graph_correct: case_type = 'Text_Only' # Text expert is better
                    else: case_type = 'Graph_Only' # Graph expert is better

                    results.append({
                        'node_idx': batch_nodes[i].item(),
                        'label': batch_labels[i].item(),
                        'case_type': case_type,
                        'pred_full': pred_full[i].item(),
                        'conf_full': conf_full[i].item(),
                        'gate': out['gate_value'][i].item(),
                        'u_text': out['u_text'][i].item(),
                        'u_graph': out['u_graph'][i].item(), # Real U_Graph
                        'conf_text': conf_text[i].item(),
                        'conf_graph': conf_graph[i].item(),
                        'homophily': self.homophily[batch_nodes[i]].item()
                    })
        
        df = pd.DataFrame(results)
        
        # --- Print Summary Report ---
        if not df.empty:
            print(f"\n📊 [Diagnosis Summary - {split.upper()}]")
            summary = df.groupby('case_type').agg({
                'node_idx': 'count',
                'gate': 'mean',
                'u_text': 'mean',
                'conf_graph': 'mean',
                'is_correct_full': 'mean' # Accuracy of fusion model on these subsets
            }).rename(columns={'node_idx': 'Count', 'is_correct_full': 'Fusion_Acc'})
            
            print(summary)
            print(f"\nMean Gate: {df['gate'].mean():.4f} | Std: {df['gate'].std():.4f}")
            
                
        return df

    # =========================================================================
    #  🎛️ Stage Control Methods
    # =========================================================================

    def pretrain_text_vib(self, epochs=15):

        print(f"\n [PreTrainer] Text Expert (VIB) Pre-training ({epochs} epochs)")

        self.pretrain_gnn = False # We are not training GNN
        self.pretrain_llm = True  # Flag for logging (optional)
        
        # Freeze Everything EXCEPT VIB
        self.freeze_module(self.gnn, freeze=True)
        self.freeze_module(self.fusion.gnn_proj, freeze=True)
        self.freeze_module(self.fusion.gate_net, freeze=True)
        self.freeze_module(self.fusion.classifier, freeze=True)
        self.freeze_module(self.fusion.norm, freeze=True)
        self.freeze_module(self.fusion.vib, freeze=False)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.fusion.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Training Loop
        best_u_gap = 0.0
        
        for epoch in range(1, epochs + 1):
            self.fusion.vib.train()
            total_loss = 0
            correct = 0
            total = 0

            u_correct_list = []
            u_wrong_list = []
            
            # KL Warmup logic for VIB
            if epoch < 5:
                kl_weight = 0.0
            else:
                progress = min(1.0, (epoch - 5) / 5)
                kl_weight = 0.01 * progress 
            
            for batch in self.dataloader:
                if isinstance(batch, (list, tuple)):
                    batch_nodes, batch_labels = batch
                else:
                    batch_nodes = batch
                    batch_labels = self.labels[batch_nodes]
                
                batch_nodes = batch_nodes.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Get Embeddings
                h_lm = self.embeddings[batch_nodes]
                
                # VIB Forward
                logits_lm, _, alpha_text, u_text, kl_loss = self.fusion.vib(h_lm)
                
                # Losses
                loss_cls = self.criterion_cls(logits_lm, batch_labels)
                if hasattr(self.edl_loss, 'epoch_num'): self.edl_loss.epoch_num = epoch
                loss_edl = self.edl_loss(alpha_text, batch_labels)
                loss_vib = kl_loss * kl_weight
                
                loss = loss_cls + loss_edl + loss_vib
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                with torch.no_grad():
                    preds = logits_lm.argmax(dim=1)
                    is_correct = (preds == batch_labels)
                    correct += is_correct.sum().item()
                    total += batch_labels.size(0)
                    
                    u_val = u_text.detach()
                    if u_val.dim() > 1: u_val = u_val.view(-1)
                    
                    u_correct_list.extend(u_val[is_correct].tolist())
                    u_wrong_list.extend(u_val[~is_correct].tolist())
                
                # Metrics
            train_acc = correct / total
            avg_loss = total_loss / len(self.dataloader)
            val_acc = self._validate_text_only()
            
            avg_u_c = sum(u_correct_list) / (len(u_correct_list) + 1e-9)
            avg_u_w = sum(u_wrong_list) / (len(u_wrong_list) + 1e-9)
            avg_u_gap = avg_u_w - avg_u_c

            
            print(f"   VIB Ep {epoch:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"   U_Corr: {avg_u_c:.4f} | U_Wrong: {avg_u_w:.4f} | Gap: {avg_u_gap:.4f}")
            
            # Validation (Text Only)
            if self.use_wandb:
                wandb.log({
                    "vib/loss": avg_loss,
                    "vib/train_acc": train_acc,
                    "vib/val_acc": val_acc,
                    "vib/u_correct": avg_u_c,
                    "vib/u_wrong": avg_u_w,
                    "vib/u_gap": avg_u_gap
                })
                
            # Save if best
            if avg_u_gap > best_u_gap:
                best_u_gap = avg_u_gap
                # Save specifically the VIB state dict
                torch.save(self.fusion.vib.state_dict(), self.ckpt_filepath.parent / "best_text_vib.pt")
        
        print(f"Text Expert Finished.")
        self.freeze_module(self.fusion.vib, freeze=True)

    def _validate_text_only(self):
        """Helper for VIB validation"""
        self.fusion.vib.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    nodes, labels = batch
                else:
                    nodes = batch
                    labels = self.labels[nodes]
                    
                nodes = nodes.to(self.device)
                h_lm = self.embeddings[nodes]
                
                logits, _, _, _, _ = self.fusion.vib(h_lm)
                preds.append(logits.argmax(dim=1).cpu())
                targets.append(labels.cpu())
                
        return accuracy_score(torch.cat(targets), torch.cat(preds))

    def pretrain_gnn_stage(self, epochs):
        print(f"\nStage 2 | GNN Pre-training ({epochs} epochs)")
        self.pretrain_gnn = True
        self.freeze_module(self.fusion, True)
        self.freeze_module(self.gnn, False)
        self._setup_optimizer()
        
        best_f1 = 0
        for ep in range(1, epochs+1):
            loss = self._train_epoch_gnn(ep)
            f1, acc = self.evaluate('val')
            print(f"GNN Ep {ep:02d} | Loss: {loss:.4f} | Val F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                self.save_checkpoint(ep)

    def pretrain_gate_stage(self, epochs):
        print(f"\nStage 3 | Gate Warmup ({epochs} epochs)")
        self.pretrain_gnn = False
        self.freeze_module(self.gnn, True)
        self.freeze_module(self.fusion.vib, True)
        self.freeze_module(self.fusion.gnn_proj, True)
        self.freeze_module(self.fusion.gate_net, False)
        self.freeze_module(self.fusion.classifier, False)
        self.freeze_module(self.fusion.norm, False)
        
        self._setup_optimizer()
        
        best_f1 = 0
        for ep in range(1, epochs+1):
            loss = self._train_epoch_fusion(ep)
            f1, acc = self.evaluate('val')
            gate_val = getattr(self, 'last_gate_mean', 0.5)
            print(f"Gate Ep {ep:02d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | Gate: {gate_val:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                self.save_checkpoint(ep)

    def train(self):
        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(epoch)
            val_f1, val_acc = self.evaluate('val')
            # append diagnostics if available
            extra = ""
            if hasattr(self, 'last_epoch_gate'):
                extra += f" | Gate: {self.last_epoch_gate:.4f}"
            if hasattr(self, 'last_epoch_u_gap'):
                extra += f" | U_Gap: {self.last_epoch_u_gap:.4f}"
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f}{extra}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_checkpoint(epoch)
        return best_f1

    # =========================================================================
    #  📉 Loss Utils
    # =========================================================================

    def calculate_uncertainty_margin_loss(self, logits, labels, uncertainty, u_low, u_high):
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        if uncertainty.dim() > 1: uncertainty = uncertainty.squeeze()
        mask_correct = (preds == labels)
        
        loss = torch.tensor(0.0, device=self.device)

        if mask_correct.sum() > 0:
            loss += F.relu(uncertainty[mask_correct] - u_low).mean()
        if (~mask_correct).sum() > 0:
            loss += F.relu(u_high - uncertainty[~mask_correct]).mean()
        return loss

    def calculate_avuc_loss(self, logits, labels, uncertainty):
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        
        if uncertainty.dim() == 1: uncertainty = uncertainty.unsqueeze(1)
        if correct.dim() == 1: correct = correct.unsqueeze(1)

        uncertainty = torch.clamp(uncertainty, min=1e-6, max=1.0 - 1e-6)

        # [FIX] Ensure target_uncertainty matches input shape [B, 1]
        ac = correct * (1 - uncertainty)
        iu = (1 - correct) * uncertainty

        loss = -torch.log(torch.mean(ac + iu) + 1e-10)
            
        return loss

    # =========================================================================
    #  💾 IO Utils
    # =========================================================================

    def save_checkpoint(self, epoch):
        self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            'fusion_state_dict': self.fusion.state_dict()
        }
        torch.save(state, self.ckpt_filepath)

    def load_checkpoint(self, path=None):
        p = path or self.ckpt_filepath
        print(f"📥 Loading checkpoint from {p}")
        try:
            state = torch.load(p, map_location=self.device)
            # 兼容性加载：检查 keys
            if 'gnn_state_dict' in state:
                self.gnn.load_state_dict(state['gnn_state_dict'])
            if 'fusion_state_dict' in state:
                self.fusion.load_state_dict(state['fusion_state_dict'])
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    def _update_monitor(self, monitor, out, labels):
        with torch.no_grad():
            preds = out['logits_lm'].argmax(1)
            is_correct = (preds == labels)
            u = out['u_text'].detach()
            monitor['u_corr'].extend(u[is_correct].tolist())
            monitor['u_wrong'].extend(u[~is_correct].tolist())
            monitor['gate'].extend(out['gate_value'].flatten().tolist())

    def _log_wandb(self, loss, loss_dict, monitor):
        u_c = np.mean(monitor['u_corr'][-100:]) if monitor['u_corr'] else 0
        u_w = np.mean(monitor['u_wrong'][-100:]) if monitor['u_wrong'] else 0
        wandb.log({
            "loss/total": loss.item(),
            "loss/cls": loss_dict['cls'].item(),
            "loss/avuc": loss_dict['avuc'].item(),
            "loss/edl_g": loss_dict['edl_g'].item(),
            "diag/u_gap": u_w - u_c
        })

    def _print_epoch_report(self, epoch, monitor):
        u_c = np.mean(monitor['u_corr']) if monitor['u_corr'] else 0
        u_w = np.mean(monitor['u_wrong']) if monitor['u_wrong'] else 0
        gate = np.mean(monitor['gate']) if monitor['gate'] else 0
        print(f"📊 [Ep {epoch}] Gate: {gate:.4f} | U_Gap: {u_w - u_c:.4f} (C:{u_c:.4f} vs W:{u_w:.4f})")