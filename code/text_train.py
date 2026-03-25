"""
text_train.py
Standalone trainer for the text-only baseline surface.

Primary baselines to keep in mind while reading this file:
- `g1_plain`: final semantic head with standard logits/probability output
- `g6_vib_edl`: VIB + EDL baseline used for stronger uncertainty analysis

The remaining matrix groups are retained as appendix-style ablations so they
can still be reproduced without being mistaken for the repository's mainline.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from text_model import TextOnlyClassifier


def _wandb_active() -> bool:
    return getattr(wandb, "run", None) is not None


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def r_edl_loss(alpha: torch.Tensor, target: torch.Tensor, num_classes: int = 2):
    """R-EDL Loss: squared error + variance term."""
    y = F.one_hot(target, num_classes=num_classes).float()
    s = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / s
    err = torch.sum((y - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)
    return torch.mean(err + var)


def edl_mse_loss_per_sample(alpha: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Original EDL paper fit term: squared error + variance, per sample."""
    y = F.one_hot(target, num_classes=num_classes).float()
    s = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / s
    err = torch.sum((y - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (s - alpha) / (s * s * (s + 1.0)), dim=1, keepdim=True)
    return err + var


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels).float()
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = confs[mask].mean().item()
        ece += mask.float().mean().item() * abs(bin_acc - bin_conf)
    return ece


def compute_ece_from_confidence(confs: torch.Tensor, correctness: torch.Tensor, n_bins: int = 15) -> float:
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confs.device, dtype=confs.dtype)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean().item()
        bin_conf = confs[mask].mean().item()
        ece += mask.float().mean().item() * abs(bin_acc - bin_conf)
    return ece


def compute_brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    y_one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    return ((probs - y_one_hot) ** 2).sum(dim=1).mean().item()


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    log_probs = torch.log(probs.clamp(min=1e-8))
    return F.nll_loss(log_probs, labels).item()


def compute_aurc(confs: torch.Tensor, correctness: torch.Tensor) -> float:
    n = confs.size(0)
    if n == 0:
        return 0.0
    order = confs.argsort(descending=True)
    sorted_correct = correctness[order]
    cum_correct = sorted_correct.cumsum(0)
    coverages = torch.arange(1, n + 1, dtype=torch.float)
    risks = 1.0 - cum_correct / coverages
    return risks.mean().item()


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return probs.clamp(min=1e-8, max=1.0 - 1e-8)


def kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL divergence between Dir(alpha) and uniform Dir(1)."""
    num_classes = alpha.size(1)
    ones = torch.ones((1, num_classes), device=alpha.device, dtype=alpha.dtype)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_ones = ones.sum(dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_ones = torch.sum(torch.lgamma(ones), dim=1, keepdim=True) - torch.lgamma(sum_ones)

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = lnB_alpha + lnB_ones + torch.sum((alpha - ones) * (digamma_alpha - digamma_sum), dim=1, keepdim=True)
    return kl


def kl_divergence_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    return kl_dirichlet(alpha)


def avuc_loss(logits: torch.Tensor, labels: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Accuracy-vs-Uncertainty calibration loss (AvUC)."""
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    correct = (preds == labels).float()
    uncertainty = uncertainty.view(-1)
    confidence = 1.0 - uncertainty
    pred_probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
    avu = (pred_probs * confidence) + ((1 - pred_probs) * (1 - confidence))
    return -torch.log(avu.clamp(min=1e-6)).mean()


class TextTrainer:
    """Text-only training/evaluation loop with calibration and per-node diagnostics."""

    def __init__(
        self,
        q_final: torch.Tensor,
        labels: torch.Tensor,
        data_dict: Dict[str, torch.Tensor],
        model: TextOnlyClassifier,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        batch_size: int,
        cfg: Dict,
        ckpt_filepath: str = "best_text.pt",
        struct_feats: Optional[torch.Tensor] = None,
        loss_mode: str = "decoupled",
        lambda_u: float = 0.2,
        eval_prob_source: str = "auto",
        early_stop_patience: int = 0,
        min_delta: float = 1e-4,
        lambda_u_warmup_epochs: int = 0,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
        vib_warmup_epochs: int = 5,
        vib_anneal_epochs: int = 10,
        vib_kl_weight: float = 1e-4,
        vib_logit_l2: float = 1e-2,
        vib_dir_kl: float = 5e-2,
        vib_mislead: float = 0.0,
        vib_avuc: float = 0.0,
        vib_scale_max: float = 3.0,
        checkpoint_metric: str = "nll",
        detach_semantics_for_u: bool = True,
        valid_split_seed: int = 42,
    ):
        self.device = device
        self.epochs = int(epochs)
        self.cfg = cfg or {}
        self.ckpt_filepath = Path(ckpt_filepath)
        self.loss_mode = loss_mode
        self.lambda_u_target = float(lambda_u)
        self.early_stop_patience = int(max(0, early_stop_patience))
        self.min_delta = float(max(0.0, min_delta))
        self.lambda_u_warmup_epochs = int(max(0, lambda_u_warmup_epochs))
        self.checkpoint_metric = str(checkpoint_metric).lower()
        self.detach_semantics_for_u = bool(detach_semantics_for_u)
        self.valid_split_seed = int(valid_split_seed)

        self.q_final = q_final.float().cpu()
        self.struct_feats = struct_feats.float().cpu() if struct_feats is not None else None
        self.labels_cpu = labels.long().cpu()
        self.labels = self.labels_cpu.to(device)
        self.num_classes = int(self.labels_cpu.max().item()) + 1

        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._maximize_checkpoint_metric = self.checkpoint_metric != "nll"
        self.scheduler = None
        if int(scheduler_patience) > 0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if self._maximize_checkpoint_metric else "min",
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
                min_lr=float(min_lr),
            )

        self.train_loader = self._build_loader(data_dict["train_idx"], batch_size, shuffle=True)
        valid_ckpt_idx, valid_cal_idx = self._resolve_validation_splits(data_dict)
        self.val_ckpt_loader = self._build_loader(valid_ckpt_idx, batch_size, shuffle=False)
        self.val_cal_loader = self._build_loader(valid_cal_idx, batch_size, shuffle=False)
        self.val_loader = self.val_ckpt_loader
        self.test_loader = self._build_loader(data_dict["test_idx"], batch_size, shuffle=False)

        self.eval_prob_source = self._resolve_eval_source(eval_prob_source)
        if self.loss_mode == "vib_edl":
            self.eval_prob_source = "alpha"

        self.vib_enabled = self.loss_mode == "vib_edl"
        self.vib_warmup_epochs = int(max(0, vib_warmup_epochs))
        self.vib_anneal_epochs = int(max(1, vib_anneal_epochs))
        self.vib_kl_weight = float(vib_kl_weight)
        self.vib_logit_l2 = float(vib_logit_l2)
        self.vib_dir_kl = float(vib_dir_kl)
        self.vib_mislead = float(vib_mislead)
        self.vib_avuc = float(vib_avuc)
        self.vib_scale_max = float(vib_scale_max)

        self.train_class_weights = self._build_class_weights(data_dict["train_idx"])

    def _build_loader(self, idx: torch.Tensor, batch_size: int, shuffle: bool):
        idx = idx.long().cpu()
        return DataLoader(
            TensorDataset(idx),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def _split_valid_idx(valid_idx: torch.Tensor, split_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_idx = valid_idx.long().cpu()
        n = int(valid_idx.numel())
        if n < 2:
            raise ValueError("valid_idx must contain at least 2 samples for valid_ckpt/valid_cal split.")
        # Text-only experiments intentionally split validation into checkpoint
        # selection and post-hoc calibration subsets. This is the cleaner
        # protocol already used by the baseline matrix.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(split_seed))
        perm = torch.randperm(n, generator=generator)
        split = max(1, min(n // 2, n - 1))
        valid_ckpt_idx = valid_idx[perm[:split]].clone()
        valid_cal_idx = valid_idx[perm[split:]].clone()
        return valid_ckpt_idx, valid_cal_idx

    def _resolve_validation_splits(self, data_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_ckpt_idx = data_dict.get("valid_ckpt_idx", None)
        valid_cal_idx = data_dict.get("valid_cal_idx", None)
        if valid_ckpt_idx is None or valid_cal_idx is None:
            valid_ckpt_idx, valid_cal_idx = self._split_valid_idx(
                data_dict["valid_idx"],
                split_seed=self.valid_split_seed,
            )
        else:
            valid_ckpt_idx = valid_ckpt_idx.long().cpu()
            valid_cal_idx = valid_cal_idx.long().cpu()
        overlap = set(valid_ckpt_idx.tolist()) & set(valid_cal_idx.tolist())
        if overlap:
            raise ValueError("valid_ckpt_idx and valid_cal_idx must be disjoint.")
        return valid_ckpt_idx, valid_cal_idx

    def _resolve_eval_source(self, source: str) -> str:
        if source == "auto":
            return "logits" if self.loss_mode == "plain" else "alpha"
        return source

    @staticmethod
    def _checkpoint_score(metrics: Dict[str, float]) -> float:
        f1 = metrics.get("f1", 0.0)
        ece = metrics.get("ece", 0.0)
        aurc = metrics.get("aurc", 0.0)
        nll = metrics.get("nll", 0.0)
        return float(f1 - 0.05 * ece - 0.05 * aurc - 0.02 * nll)

    def _select_checkpoint_value(self, metrics: Dict[str, float]) -> float:
        mode = self.checkpoint_metric
        if mode == "acc":
            return float(metrics.get("acc", 0.0))
        if mode == "f1":
            return float(metrics.get("f1", 0.0))
        if mode == "nll":
            return float(metrics.get("nll", float("inf")))
        if mode == "score":
            return self._checkpoint_score(metrics)
        raise ValueError(f"Unsupported checkpoint_metric: {self.checkpoint_metric}")

    def _alpha_for_u_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not (self.loss_mode == "decoupled" and self.detach_semantics_for_u):
            return out["alpha_text"]
        if not bool(out.get("alpha_from_logits", False)):
            return out["alpha_text"]

        # True decoupling: CE updates semantic logits; EDL updates uncertainty heads.
        logits_detached = out["logits_text"].detach()
        temp = out["temperature_text"]
        concentration = out["concentration_text"]
        probs_u = F.softmax(logits_detached / temp, dim=1)
        return probs_u * concentration + 1.0

    def _current_lambda_u(self, epoch: int) -> float:
        if self.loss_mode != "decoupled":
            return self.lambda_u_target
        if self.lambda_u_warmup_epochs <= 0:
            return self.lambda_u_target
        ratio = min(1.0, max(0.0, epoch / float(self.lambda_u_warmup_epochs)))
        return self.lambda_u_target * ratio

    def _loader_for_split(self, split: str):
        if split in {"val", "val_ckpt"}:
            return self.val_ckpt_loader
        if split == "val_cal":
            return self.val_cal_loader
        if split == "test":
            return self.test_loader
        raise ValueError(f"Unsupported split: {split}")

    def _compute_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, lambda_u_now: float, epoch: Optional[int] = None):
        if self.loss_mode == "vib_edl":
            return self._compute_vib_loss(out, y, epoch)

        loss_sem = F.cross_entropy(out["logits_text"], y)
        alpha_for_u = self._alpha_for_u_loss(out)
        loss_u = r_edl_loss(alpha_for_u, y, num_classes=self.num_classes)

        if self.loss_mode == "plain":
            loss = loss_sem
        elif self.loss_mode == "edl":
            loss = loss_u
        elif self.loss_mode == "decoupled":
            loss = loss_sem + float(lambda_u_now) * loss_u
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

        return loss, {
            "loss_total": float(loss.item()),
            "loss_sem": float(loss_sem.item()),
            "loss_u": float(loss_u.item()),
            "lambda_u_now": float(lambda_u_now),
        }

    def _compute_vib_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor, epoch: Optional[int]):
        logits = out["logits_text"]
        alpha = out["alpha_text"]
        vib_kl = out.get("vib_kl", torch.zeros(1, device=logits.device))
        sample_weights = self.train_class_weights[y]
        is_warmup = (epoch or 0) <= self.vib_warmup_epochs
        loss_dict = {}

        if is_warmup:
            loss_cls = F.cross_entropy(logits, y, weight=self.train_class_weights)
            loss_logit_l2 = torch.mean(logits ** 2)
            loss = loss_cls + self.vib_logit_l2 * loss_logit_l2 + self.vib_kl_weight * vib_kl.mean()
            loss_dict["loss_sem"] = float(loss_cls.item())
            loss_dict["loss_u"] = 0.0
            loss_dict["loss_ce"] = float(loss_cls.item())
            loss_dict["loss_logit_l2"] = float(loss_logit_l2.item())
            loss_dict["loss_vib_kl"] = float(vib_kl.mean().item())
        else:
            y_hot = F.one_hot(y, num_classes=self.num_classes).float()
            fit_terms = edl_mse_loss_per_sample(alpha, y, num_classes=self.num_classes)
            loss_fit = torch.mean(fit_terms * sample_weights.unsqueeze(1))
            adjusted_alpha = y_hot + (1.0 - y_hot) * alpha
            kl_dir = kl_dirichlet(adjusted_alpha)
            anneal_progress = (epoch - self.vib_warmup_epochs) / max(1, self.vib_anneal_epochs)
            anneal_factor = min(1.0, max(0.0, anneal_progress))
            loss_dir = self.vib_dir_kl * anneal_factor * torch.mean(kl_dir * sample_weights.unsqueeze(1))

            # Keep optional extras available, but the default VIB-EDL path is paper-aligned.
            evidence = alpha - 1.0
            non_target = 1.0 - y_hot
            loss_mislead = torch.zeros(1, device=logits.device)
            if self.vib_mislead > 0.0:
                loss_mislead = self.vib_mislead * anneal_factor * torch.mean(
                    torch.sum(evidence * non_target * sample_weights.unsqueeze(1), dim=1)
                )

            loss_avuc = torch.zeros(1, device=logits.device)
            if self.vib_avuc > 0.0:
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
                logits_for_avuc = torch.log(probs + 1e-8)
                loss_avuc = self.vib_avuc * anneal_factor * avuc_loss(logits_for_avuc, y, out["u_text"])

            loss = loss_fit + loss_dir + self.vib_kl_weight * vib_kl.mean() + loss_mislead + loss_avuc
            loss_dict["loss_sem"] = float(loss_fit.item())
            loss_dict["loss_u"] = float((loss_dir + loss_mislead + loss_avuc).item())
            loss_dict.update({
                "loss_edl_fit": float(loss_fit.item()),
                "loss_dir_kl": float(loss_dir.item()),
                "loss_mislead": float(loss_mislead.item()),
                "loss_avuc": float(loss_avuc.item()),
                "loss_vib_kl": float(vib_kl.mean().item()),
            })

        loss_dict["loss_total"] = float(loss.item())
        loss_dict["loss_vib_phase"] = float(is_warmup)
        loss_dict["lambda_u_now"] = 0.0
        return loss, loss_dict

    def _probs_from_out(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.eval_prob_source == "alpha":
            return out["prob_text"]
        if self.eval_prob_source == "logits":
            return F.softmax(out["logits_text"], dim=1)
        raise ValueError(f"Unsupported eval_prob_source: {self.eval_prob_source}")

    def _clamp_evidence_scale(self) -> None:
        branch = getattr(self.model, "text_branch", None)
        if branch is None:
            return
        scale = getattr(branch, "evidence_scale", None)
        if scale is None:
            return
        with torch.no_grad():
            scale.clamp_(max=self.vib_scale_max)

    def _logits_for_ts(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.eval_prob_source == "alpha":
            return torch.log(out["prob_text"].clamp(min=1e-8))
        if self.eval_prob_source == "logits":
            return out["logits_text"]
        raise ValueError(f"Unsupported eval_prob_source: {self.eval_prob_source}")

    @staticmethod
    def _compute_prob_metrics(
        probs_cat: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
        decision_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        metrics = {
            "f1": float(f1_score(targets, preds, average="macro")),
            "acc": float(accuracy_score(targets, preds)),
        }
        confs = decision_conf if decision_conf is not None else probs_cat.max(dim=1).values
        correctness = (preds == targets).float()
        metrics["ece_10"] = float(compute_ece_from_confidence(confs, correctness, n_bins=10))
        metrics["ece_15"] = float(compute_ece_from_confidence(confs, correctness, n_bins=15))
        metrics["ece_20"] = float(compute_ece_from_confidence(confs, correctness, n_bins=20))
        metrics["ece"] = float(metrics["ece_15"])
        metrics["brier"] = float(compute_brier(probs_cat, targets))
        metrics["nll"] = float(compute_nll(probs_cat, targets))
        metrics["aurc"] = float(compute_aurc(confs, correctness))
        return metrics

    def _normalize_calibrators(self, calibrators: Optional[List[str]]) -> List[str]:
        if not calibrators:
            return []
        supported = {"ts", "platt", "beta", "isotonic"}
        dedup = []
        seen = set()
        for c in calibrators:
            c_norm = str(c).strip().lower()
            if c_norm in seen:
                continue
            if c_norm not in supported:
                raise ValueError(f"Unsupported calibrator: {c_norm}. Supported: {sorted(supported)}")
            dedup.append(c_norm)
            seen.add(c_norm)
        return dedup

    @staticmethod
    def _binary_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.size(1) != 2:
            raise ValueError("Binary calibrators require exactly 2 classes.")
        return (logits[:, 1] - logits[:, 0]).view(-1, 1)

    @staticmethod
    def _probs_from_pos_binary(p1: np.ndarray) -> torch.Tensor:
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
        p1 = np.clip(p1, 1e-8, 1.0 - 1e-8)
        p0 = 1.0 - p1
        return torch.from_numpy(np.stack([p0, p1], axis=1)).float()

    def _fit_single_calibrator(self, name: str, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        name = str(name).lower()
        logits = logits.float()
        labels = labels.long()
        if logits.numel() == 0:
            raise ValueError("Empty logits for calibrator fitting.")
        if name == "ts":
            t_param = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.LBFGS([t_param], lr=0.01, max_iter=50)

            def closure():
                optimizer.zero_grad()
                scaled = logits / t_param.clamp(min=0.1)
                loss = F.cross_entropy(scaled, labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            return {"name": "ts", "temperature": float(t_param.clamp(min=0.1).item())}

        score = self._binary_score_from_logits(logits).cpu().numpy()
        y = labels.cpu().numpy().astype(np.int64)

        if name == "platt":
            model = LogisticRegression(C=1e6, solver="lbfgs", fit_intercept=True, max_iter=2000)
            model.fit(score, y)
            return {"name": "platt", "model": model}

        if name == "beta":
            p = 1.0 / (1.0 + np.exp(-score.reshape(-1)))
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            x = np.stack([np.log(p), -np.log(1.0 - p)], axis=1)
            model = LogisticRegression(C=1e6, solver="lbfgs", fit_intercept=True, max_iter=2000)
            model.fit(x, y)
            return {"name": "beta", "model": model}

        if name == "isotonic":
            p = 1.0 / (1.0 + np.exp(-score.reshape(-1)))
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p, y.astype(np.float64))
            return {"name": "isotonic", "model": model}

        raise ValueError(f"Unsupported calibrator: {name}")

    def _apply_single_calibrator(self, state: Dict[str, Any], logits: torch.Tensor) -> torch.Tensor:
        name = state["name"]
        logits = logits.float()
        if name == "ts":
            temp = max(float(state["temperature"]), 0.1)
            return _safe_softmax(logits / temp)

        score = self._binary_score_from_logits(logits).cpu().numpy()
        if name == "platt":
            p1 = state["model"].predict_proba(score)[:, 1]
            return self._probs_from_pos_binary(p1)

        if name == "beta":
            p = 1.0 / (1.0 + np.exp(-score.reshape(-1)))
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            x = np.stack([np.log(p), -np.log(1.0 - p)], axis=1)
            p1 = state["model"].predict_proba(x)[:, 1]
            return self._probs_from_pos_binary(p1)

        if name == "isotonic":
            p = 1.0 / (1.0 + np.exp(-score.reshape(-1)))
            p = np.clip(p, 1e-8, 1.0 - 1e-8)
            p1 = state["model"].predict(p)
            return self._probs_from_pos_binary(p1)

        raise ValueError(f"Unsupported calibrator state: {name}")

    @staticmethod
    def _calibrator_state_to_params(state: Dict[str, Any]) -> Dict[str, Any]:
        name = str(state.get("name", "unknown"))
        if name == "ts":
            return {"name": "ts", "temperature": float(state.get("temperature", 1.0))}
        if name in {"platt", "beta"}:
            model = state.get("model", None)
            if model is None:
                return {"name": name}
            return {
                "name": name,
                "coef": model.coef_.reshape(-1).astype(float).tolist(),
                "intercept": model.intercept_.reshape(-1).astype(float).tolist(),
            }
        if name == "isotonic":
            model = state.get("model", None)
            if model is None:
                return {"name": "isotonic"}
            return {
                "name": "isotonic",
                "n_thresholds": int(len(model.X_thresholds_)),
                "x_thresholds": np.asarray(model.X_thresholds_).astype(float).tolist(),
                "y_thresholds": np.asarray(model.y_thresholds_).astype(float).tolist(),
            }
        return {"name": name}

    def _build_class_weights(self, train_idx: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(self.labels_cpu[train_idx.long()], minlength=self.num_classes).float()
        total = counts.sum()
        weights = total / (counts * self.num_classes)
        weights[torch.isinf(weights) | torch.isnan(weights)] = 1.0
        return weights.to(self.device)

    def _train_epoch(self, epoch: int):
        self.model.train()
        agg = defaultdict(list)
        lambda_u_now = self._current_lambda_u(epoch)

        for (idx_cpu,) in self.train_loader:
            idx = idx_cpu.to(self.device)
            x = self.q_final[idx_cpu].to(self.device)
            y = self.labels[idx]

            self.optimizer.zero_grad()
            out = self.model(x, self.cfg.get("text"))
            loss, logs = self._compute_loss(out, y, lambda_u_now=lambda_u_now, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            self._clamp_evidence_scale()

            for k, v in logs.items():
                agg[k].append(v)

        out_logs = {k: float(np.mean(v)) for k, v in agg.items()}
        out_logs.setdefault("loss_sem", out_logs.get("loss_total", 0.0))
        out_logs.setdefault("loss_u", 0.0)
        out_logs["lambda_u_now"] = float(lambda_u_now)
        return out_logs

    @torch.no_grad()
    def evaluate(
        self,
        split: str = "val",
        save_per_node: bool = False,
        save_name: Optional[str] = None,
    ):
        self.model.eval()
        loader = self._loader_for_split(split)

        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []
        stats = {
            "u_text": [],
            "temperature_text": [],
            "concentration_text": [],
        }
        per_node_records: List[Dict] = []

        for (idx_cpu,) in loader:
            idx_dev = idx_cpu.to(self.device)
            x = self.q_final[idx_cpu].to(self.device)
            y = self.labels[idx_dev]

            out = self.model(x, self.cfg.get("text"))
            prob = self._probs_from_out(out)
            preds = prob.argmax(dim=-1)
            bsz = int(idx_cpu.size(0))

            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

            stats["u_text"].extend(out["u_text"].view(-1).detach().cpu().tolist())
            stats["temperature_text"].extend(
                out["temperature_text"].view(-1).detach().cpu().tolist()
            )
            stats["concentration_text"].extend(
                out["concentration_text"].view(-1).detach().cpu().tolist()
            )

            if save_per_node:
                prob_cpu = prob.detach().cpu()
                u_text = out["u_text"].detach().cpu()
                t_text = out["temperature_text"].detach().cpu()
                c_text = out["concentration_text"].detach().cpu()
                logits = out["logits_text"].detach().cpu()
                y_cpu = y.detach().cpu()
                pred_cpu = preds.detach().cpu()

                for i in range(bsz):
                    node_id = int(idx_cpu[i].item())
                    rec = {
                        "node_id": node_id,
                        "y": int(y_cpu[i].item()),
                        "pred_text": int(pred_cpu[i].item()),
                        "correct_text": int(pred_cpu[i].item() == y_cpu[i].item()),
                        "conf_text": float(prob_cpu[i].max().item()),
                        "prob_text_0": float(prob_cpu[i, 0].item()),
                        "prob_text_1": float(prob_cpu[i, 1].item()) if prob_cpu.size(1) > 1 else 0.0,
                        "u_text": float(u_text[i].item()),
                        "temperature_text": float(t_text[i].item()),
                        "concentration_text": float(c_text[i].item()),
                        "logit_text_0": float(logits[i, 0].item()),
                        "logit_text_1": float(logits[i, 1].item()) if logits.size(1) > 1 else 0.0,
                    }
                    if self.struct_feats is not None:
                        sf = self.struct_feats[node_id]
                        rec.update({
                            "in_log_degree": float(sf[0].item()),
                            "out_log_degree": float(sf[1].item()),
                            "total_log_degree": float(sf[2].item()),
                            "graph_missing": float(sf[4].item()) if sf.size(0) > 4 else 0.0,
                        })
                    per_node_records.append(rec)

        p = torch.cat(all_preds)
        t = torch.cat(all_targets)
        probs_cat = torch.cat(all_probs)

        metrics = self._compute_prob_metrics(probs_cat, t, p)

        metrics.update({k: float(np.mean(v)) for k, v in stats.items() if len(v) > 0})

        if save_per_node and per_node_records:
            file_name = save_name or f"per_node_{split}_text.jsonl"
            dump_path = self.ckpt_filepath.parent / file_name
            with open(dump_path, "w", encoding="utf-8") as f:
                for rec in per_node_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Saved text per-node diagnostics to {dump_path}")

        return metrics

    @torch.no_grad()
    def _collect_logits(self, loader):
        self.model.eval()
        all_logits = []
        all_labels = []
        for (idx_cpu,) in loader:
            idx = idx_cpu.to(self.device)
            x = self.q_final[idx_cpu].to(self.device)
            y = self.labels[idx]
            out = self.model(x, self.cfg.get("text"))
            logits = self._logits_for_ts(out)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
        return torch.cat(all_logits), torch.cat(all_labels)

    def posthoc_calibrate(self, max_iter: int = 50, lr: float = 0.01):
        logits, labels = self._collect_logits(self.val_cal_loader)
        state = self._fit_single_calibrator("ts", logits, labels)
        t_opt = float(state["temperature"])
        print(f"  Text post-hoc T = {t_opt:.4f}")
        return t_opt

    def fit_calibrators(self, calibrators: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        names = self._normalize_calibrators(calibrators)
        if not names:
            return {}
        logits, labels = self._collect_logits(self.val_cal_loader)
        states: Dict[str, Dict[str, Any]] = {}
        for name in names:
            try:
                states[name] = self._fit_single_calibrator(name, logits, labels)
            except Exception as e:
                print(f"[Warning] Failed to fit calibrator {name}: {e}")
        return states

    @torch.no_grad()
    def evaluate_with_calibrator(
        self,
        calibrator_state: Dict[str, Any],
        split: str = "test",
        save_per_node: bool = False,
        save_name: Optional[str] = None,
        preserve_decisions: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        loader = self._loader_for_split(split)
        cal_name = str(calibrator_state.get("name", "unknown"))

        all_preds, all_targets, all_probs, all_decision_conf = [], [], [], []
        per_node_records: List[Dict] = []

        for (idx_cpu,) in loader:
            idx_dev = idx_cpu.to(self.device)
            x = self.q_final[idx_cpu].to(self.device)
            y = self.labels[idx_dev]

            out = self.model(x, self.cfg.get("text"))
            logits = self._logits_for_ts(out)
            prob = self._apply_single_calibrator(calibrator_state, logits).to(self.device)
            raw_prob = self._probs_from_out(out)
            raw_preds = raw_prob.argmax(dim=-1)
            preds = raw_preds if preserve_decisions else prob.argmax(dim=-1)
            decision_conf = prob.gather(1, preds.unsqueeze(1)).squeeze(1)
            bsz = int(idx_cpu.size(0))

            all_probs.append(prob.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_decision_conf.append(decision_conf.cpu())

            if save_per_node:
                y_cpu = y.detach().cpu()
                pred_cpu = preds.detach().cpu()
                raw_pred_cpu = raw_preds.detach().cpu()
                prob_cpu = prob.detach().cpu()
                conf_cpu = decision_conf.detach().cpu()
                out_prob = raw_prob.detach().cpu()
                logits_text = out["logits_text"].detach().cpu()
                u_text = out["u_text"].detach().cpu()
                t_text = out["temperature_text"].detach().cpu()
                c_text = out["concentration_text"].detach().cpu()
                for i in range(bsz):
                    node_id = int(idx_cpu[i].item())
                    rec = {
                        "node_id": node_id,
                        "y": int(y_cpu[i].item()),
                        "pred_text": int(pred_cpu[i].item()),
                        "pred_text_raw": int(raw_pred_cpu[i].item()),
                        "correct_text": int(pred_cpu[i].item() == y_cpu[i].item()),
                        "conf_text": float(conf_cpu[i].item()),
                        "prob_text_0": float(prob_cpu[i, 0].item()),
                        "prob_text_1": float(prob_cpu[i, 1].item()) if prob_cpu.size(1) > 1 else 0.0,
                        "u_text": float(u_text[i].item()),
                        "temperature_text": float(t_text[i].item()),
                        "concentration_text": float(c_text[i].item()),
                        "logit_text_0": float(logits_text[i, 0].item()),
                        "logit_text_1": float(logits_text[i, 1].item()) if logits_text.size(1) > 1 else 0.0,
                        "calibrator": cal_name,
                        "raw_prob_text_0": float(out_prob[i, 0].item()),
                        "raw_prob_text_1": float(out_prob[i, 1].item()) if out_prob.size(1) > 1 else 0.0,
                    }
                    if cal_name == "ts":
                        rec["T_posthoc"] = float(calibrator_state.get("temperature", 1.0))
                    rec["preserve_decisions"] = int(bool(preserve_decisions))
                    if self.struct_feats is not None:
                        sf = self.struct_feats[node_id]
                        rec.update({
                            "in_log_degree": float(sf[0].item()),
                            "out_log_degree": float(sf[1].item()),
                            "total_log_degree": float(sf[2].item()),
                            "graph_missing": float(sf[4].item()) if sf.size(0) > 4 else 0.0,
                        })
                    per_node_records.append(rec)

        p = torch.cat(all_preds)
        t = torch.cat(all_targets)
        probs_cat = torch.cat(all_probs)
        decision_conf_cat = torch.cat(all_decision_conf)
        metrics = self._compute_prob_metrics(probs_cat, t, p, decision_conf=decision_conf_cat)

        if save_per_node and per_node_records:
            file_name = save_name or f"per_node_{split}_text_cal_{cal_name}.jsonl"
            dump_path = self.ckpt_filepath.parent / file_name
            with open(dump_path, "w", encoding="utf-8") as f:
                for rec in per_node_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Saved calibrated text per-node diagnostics to {dump_path}")
        return metrics

    @torch.no_grad()
    def evaluate_calibrated(
        self,
        t_posthoc: float,
        split: str = "test",
        save_per_node: bool = False,
        save_name: Optional[str] = None,
    ):
        state = {"name": "ts", "temperature": max(float(t_posthoc), 0.1)}
        return self.evaluate_with_calibrator(
            state,
            split=split,
            save_per_node=save_per_node,
            save_name=save_name,
        )

    def train(
        self,
        run_posthoc: bool = True,
        save_per_node: bool = True,
        calibrators: Optional[List[str]] = None,
        artifact_tag: str = "",
    ):
        best_score = float("inf") if not self._maximize_checkpoint_metric else -float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        suffix = f"_{artifact_tag}" if artifact_tag else ""
        print(f"\nStarting Text-only training (loss_mode={self.loss_mode})...")
        for epoch in range(1, self.epochs + 1):
            train_logs = self._train_epoch(epoch)
            val_ckpt = self.evaluate("val_ckpt", save_per_node=False)
            score = self._select_checkpoint_value(val_ckpt)
            lr_now = float(self.optimizer.param_groups[0]["lr"])
            if self.scheduler is not None:
                self.scheduler.step(score)

            if _wandb_active():
                wandb.log({
                    "epoch": epoch,
                    "text/train_loss": train_logs["loss_total"],
                    "text/train_loss_sem": train_logs["loss_sem"],
                    "text/train_loss_u": train_logs["loss_u"],
                    "text/train_lambda_u_now": train_logs["lambda_u_now"],
                    "text/lr": lr_now,
                    **{f"text/val_ckpt_{k}": v for k, v in val_ckpt.items()},
                    "text/val_ckpt_value": score,
                    "text/ckpt_metric_mode": {"acc": 0, "f1": 1, "score": 2, "nll": 3}[self.checkpoint_metric],
                    "text/loss_mode": {"plain": 0, "edl": 1, "decoupled": 2, "vib_edl": 3}[self.loss_mode],
                })

            ece_str = f" | ECE={val_ckpt['ece']:.4f}" if "ece" in val_ckpt else ""
            aurc_str = f" | AURC={val_ckpt['aurc']:.4f}" if "aurc" in val_ckpt else ""
            extra = f" | LR={lr_now:.2e}"
            if self.loss_mode == "decoupled":
                extra += f" | lambda_u={train_logs['lambda_u_now']:.4f}"
            elif self.loss_mode == "vib_edl":
                extra += f" | vib_warmup={int(epoch <= self.vib_warmup_epochs)}"
            print(
                f"Epoch {epoch:02d} | "
                f"L={train_logs['loss_total']:.4f} "
                f"(sem={train_logs['loss_sem']:.4f}, u={train_logs['loss_u']:.4f}) | "
                f"F1={val_ckpt['f1']:.4f} | Acc={val_ckpt['acc']:.4f}"
                f"{ece_str}{aurc_str} | CKPT({self.checkpoint_metric})={score:.4f}{extra}"
            )

            improved = (
                score > (best_score + self.min_delta)
                if self._maximize_checkpoint_metric
                else score < (best_score - self.min_delta)
            )
            if improved:
                best_score = score
                best_epoch = epoch
                epochs_no_improve = 0
                self.ckpt_filepath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "best_val_score": best_score,
                        "best_epoch": best_epoch,
                        "best_val_metrics": val_ckpt,
                        "checkpoint_metric": self.checkpoint_metric,
                        "detach_semantics_for_u": self.detach_semantics_for_u,
                        "loss_mode": self.loss_mode,
                        "eval_prob_source": self.eval_prob_source,
                    },
                    self.ckpt_filepath,
                )
            else:
                epochs_no_improve += 1

            if self.early_stop_patience > 0 and epochs_no_improve >= self.early_stop_patience:
                print(
                    f"  Early stopping at epoch {epoch:02d} "
                    f"(best_epoch={best_epoch:02d}, best_{self.checkpoint_metric}={best_score:.4f})"
                )
                break

        ckpt = torch.load(self.ckpt_filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "best_epoch" in ckpt:
            print(
                f"  Loaded best checkpoint from epoch {int(ckpt['best_epoch']):02d} "
                f"({self.checkpoint_metric}={float(ckpt.get('best_val_score', 0.0)):.4f})"
            )

        val_ckpt_raw = self.evaluate("val_ckpt", save_per_node=False)
        val_raw = self.evaluate(
            "val_cal",
            save_per_node=save_per_node,
            save_name=f"per_node_val_cal_text{suffix}.jsonl" if save_per_node else None,
        )
        test_raw = self.evaluate(
            "test",
            save_per_node=save_per_node,
            save_name=f"per_node_test_text{suffix}.jsonl" if save_per_node else None,
        )
        print(
            "Text Test (raw): "
            f"F1={test_raw['f1']:.4f} | Acc={test_raw['acc']:.4f}"
            + (f" | ECE={test_raw['ece']:.4f}" if "ece" in test_raw else "")
            + (f" | Brier={test_raw['brier']:.4f}" if "brier" in test_raw else "")
            + (f" | NLL={test_raw['nll']:.4f}" if "nll" in test_raw else "")
            + (f" | AURC={test_raw['aurc']:.4f}" if "aurc" in test_raw else "")
        )

        cal_names = self._normalize_calibrators(calibrators)
        if calibrators is None:
            cal_names = ["ts"] if run_posthoc else []
        cal_states = self.fit_calibrators(cal_names)

        result: Dict[str, Any] = {
            "val_ckpt_raw": val_ckpt_raw,
            "val_raw": val_raw,
            "test_raw": test_raw,
            "calibrator_params": {},
            "val_calibrated": {},
            "test_calibrated": {},
        }
        for name in cal_names:
            if name not in cal_states:
                continue
            state = cal_states[name]
            val_cal = self.evaluate_with_calibrator(
                state,
                "val_cal",
                save_per_node=save_per_node,
                save_name=f"per_node_val_cal_text_cal_{name}{suffix}.jsonl" if save_per_node else None,
            )
            test_cal = self.evaluate_with_calibrator(
                state,
                "test",
                save_per_node=save_per_node,
                save_name=f"per_node_test_text_cal_{name}{suffix}.jsonl" if save_per_node else None,
            )
            result["calibrator_params"][name] = self._calibrator_state_to_params(state)
            result["val_calibrated"][name] = val_cal
            result["test_calibrated"][name] = test_cal
            label = f"{name.upper()}"
            if name == "ts":
                label = f"T={float(state.get('temperature', 1.0)):.4f}"
            print(
                f"Text Test ({label}): "
                f"F1={test_cal['f1']:.4f} | Acc={test_cal['acc']:.4f}"
                + (f" | ECE={test_cal['ece']:.4f}" if "ece" in test_cal else "")
                + (f" | Brier={test_cal['brier']:.4f}" if "brier" in test_cal else "")
                + (f" | NLL={test_cal['nll']:.4f}" if "nll" in test_cal else "")
                + (f" | AURC={test_cal['aurc']:.4f}" if "aurc" in test_cal else "")
            )

        # Backward-compatible aliases for TS consumers.
        if "ts" in result["test_calibrated"]:
            result["posthoc_T"] = float(result["calibrator_params"]["ts"]["temperature"])
            result["test_cal"] = result["test_calibrated"]["ts"]
            if save_per_node:
                # Keep legacy file name for external scripts.
                legacy_src = self.ckpt_filepath.parent / f"per_node_test_text_cal_ts{suffix}.jsonl"
                legacy_dst = self.ckpt_filepath.parent / f"per_node_test_text_calibrated{suffix}.jsonl"
                if legacy_src.exists():
                    legacy_dst.write_text(legacy_src.read_text(encoding="utf-8"), encoding="utf-8")

        summary_path = self.ckpt_filepath.parent / f"text_metrics_summary{suffix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if _wandb_active():
            wandb.log({f"text/test_raw_{k}": v for k, v in test_raw.items()})
            for cal_name, cal_metrics in result["test_calibrated"].items():
                wandb.log({f"text/test_cal_{cal_name}_{k}": v for k, v in cal_metrics.items()})
            if "posthoc_T" in result:
                wandb.log({"text/posthoc_T": result["posthoc_T"]})

        return result


def run_text_experiment_matrix(
    q_final: torch.Tensor,
    labels: torch.Tensor,
    data_dict: Dict[str, torch.Tensor],
    struct_feats: Optional[torch.Tensor],
    device: torch.device,
    in_dim: int,
    hid_dim: int,
    num_classes: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    cfg_text: Dict,
    ckpt_root: Path,
    seed: int = 42,
    lambda_u: float = 0.2,
    same_seed_across_groups: bool = True,
    early_stop_patience: int = 0,
    min_delta: float = 1e-4,
    lambda_u_warmup_epochs: int = 0,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    checkpoint_metric: str = "nll",
    detach_semantics_for_u: bool = True,
    vib_warmup_epochs: int = 5,
    vib_anneal_epochs: int = 10,
    vib_kl_weight: float = 1e-4,
    vib_logit_l2: float = 1e-2,
    vib_dir_kl: float = 5e-2,
    vib_mislead: float = 0.0,
    vib_avuc: float = 0.0,
    vib_scale_max: float = 3.0,
    calibrators: Optional[List[str]] = None,
    valid_split_seed: int = 42,
):
    """Run the text-only ablation matrix.

    Base models are trained once (g1/g3/g4/g6). Calibrated legacy views
    (g2/g5/g7) are synthesized from the corresponding base checkpoint.

    Reading guide:
    - `g1_plain` is the primary probability baseline.
    - `g6_vib_edl` is the primary uncertainty-aware baseline.
    - Classwise variants and derived TS views are appendix comparisons.
    """
    # Base groups define train-once backbones. The paper-facing emphasis is on
    # `g1_plain` and `g6_vib_edl`; the others remain reproducible side studies.
    base_groups = [
        {
            "id": "g1_plain",
            "method": "Final semantic head",
            "loss_mode": "plain",
            "text_branch_type": "semantic",
            "default_output": "raw",
            "eval_prob_source": "logits",
        },
        {
            "id": "g3_adapter_edl",
            "method": "Final semantic + uncertainty adapter",
            "loss_mode": "edl",
            "text_branch_type": "semantic",
            "default_output": "raw",
            "eval_prob_source": "alpha",
        },
        {
            "id": "g4_decoupled",
            "method": "Decoupled CE + EDL (legacy scalar concentration)",
            "loss_mode": "decoupled",
            "text_branch_type": "semantic",
            "default_output": "raw",
            "eval_prob_source": "alpha",
        },
        {
            "id": "g8_classwise_edl",
            "method": "Class-wise evidence head",
            "loss_mode": "edl",
            "text_branch_type": "classwise_edl",
            "default_output": "raw",
            "eval_prob_source": "alpha",
        },
        {
            "id": "g9_classwise_decoupled",
            "method": "Class-wise evidence + decoupled CE",
            "loss_mode": "decoupled",
            "text_branch_type": "classwise_edl",
            "default_output": "raw",
            "eval_prob_source": "alpha",
        },
        {
            "id": "g6_vib_edl",
            "method": "VIB + EDL baseline",
            "loss_mode": "vib_edl",
            "text_branch_type": "vib_edl",
            "default_output": "raw",
            "eval_prob_source": "alpha",
        },
    ]
    # Derived groups are post-hoc views built from the base checkpoints above.
    # They should be read as calibration/reporting variants, not independent
    # architectural mainlines.
    derived_groups = {
        "g1_plain": [
            {
                "id": "g2_plain_ts",
                "method": "Final semantic head + TS",
                "loss_mode": "plain",
                "default_output": "ts",
                "eval_prob_source": "logits",
            },
        ],
        "g4_decoupled": [
            {
                "id": "g5_decoupled_ts",
                "method": "Decoupled CE + EDL + TS (legacy scalar concentration)",
                "loss_mode": "decoupled",
                "text_branch_type": "semantic",
                "default_output": "ts",
                "eval_prob_source": "alpha",
            },
        ],
        "g9_classwise_decoupled": [
            {
                "id": "g10_classwise_decoupled_ts",
                "method": "Class-wise evidence + decoupled CE + TS",
                "loss_mode": "decoupled",
                "text_branch_type": "classwise_edl",
                "default_output": "ts",
                "eval_prob_source": "alpha",
            },
        ],
        "g6_vib_edl": [
            {
                "id": "g7_vib_edl_ts",
                "method": "VIB + EDL + TS",
                "loss_mode": "vib_edl",
                "text_branch_type": "vib_edl",
                "default_output": "ts",
                "eval_prob_source": "alpha",
            },
        ],
    }

    table_rows = []
    full_rows = []
    ckpt_root.mkdir(parents=True, exist_ok=True)

    def _resolve_selected_output(outputs: Dict[str, Dict[str, float]], output_name: str) -> str:
        if output_name != "raw" and output_name in outputs:
            return output_name
        return "raw"

    def _append_group_result(
        group_spec: Dict[str, str],
        outputs: Dict[str, Dict[str, float]],
        val_outputs: Dict[str, Dict[str, float]],
        calibrator_params: Dict[str, Dict[str, Any]],
        ckpt_path_now: Path,
        run_dir: Path,
        run_seed: int,
        source_group_id: str,
        is_posthoc_view: bool,
    ) -> None:
        selected_output = _resolve_selected_output(outputs, str(group_spec.get("default_output", "raw")))
        metric = outputs.get(selected_output, outputs.get("raw", {}))
        table_rows.append({
            "group_id": group_spec["id"],
            "method": group_spec["method"],
            "f1": float(metric.get("f1", 0.0)),
            "acc": float(metric.get("acc", 0.0)),
            "ece": float(metric.get("ece", 0.0)),
            "brier": float(metric.get("brier", 0.0)),
            "nll": float(metric.get("nll", 0.0)),
            "aurc": float(metric.get("aurc", 0.0)),
            "selected_output": selected_output,
            "loss_mode": group_spec["loss_mode"],
            "ckpt_path": str(ckpt_path_now),
            "run_dir": str(run_dir),
            "seed_used": run_seed,
            "source_group_id": source_group_id,
            "is_posthoc_view": int(bool(is_posthoc_view)),
        })
        full_rows.append({
            "group_id": group_spec["id"],
            "method": group_spec["method"],
            "loss_mode": group_spec["loss_mode"],
            "seed_used": run_seed,
            "ckpt_path": str(ckpt_path_now),
            "run_dir": str(run_dir),
            "selected_output": selected_output,
            "source_group_id": source_group_id,
            "is_posthoc_view": int(bool(is_posthoc_view)),
            "outputs": outputs,
            "val_outputs": val_outputs,
            "calibrator_params": calibrator_params,
        })

    def _materialize_view_artifacts(src_dir: Path, dst_dir: Path, seed_now: int, output_names: List[str]) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        candidate_files = {
            f"per_node_test_text_seed{seed_now}.jsonl",
            f"per_node_val_cal_text_seed{seed_now}.jsonl",
            f"text_metrics_summary_seed{seed_now}.json",
            f"per_node_test_text_calibrated_seed{seed_now}.jsonl",
        }
        for output_name in output_names:
            if output_name == "raw":
                continue
            candidate_files.add(f"per_node_test_text_cal_{output_name}_seed{seed_now}.jsonl")
            candidate_files.add(f"per_node_val_cal_text_cal_{output_name}_seed{seed_now}.jsonl")
        for name in sorted(candidate_files):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)

    for i, g in enumerate(base_groups):
        run_seed = int(seed if same_seed_across_groups else (seed + i))
        _set_global_seed(run_seed)

        run_dir = ckpt_root / g["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / f"best_text_seed{seed}.pt"

        print("\n" + "=" * 80)
        print(f"[Text Matrix] {g['id']} | {g['method']}")
        print("=" * 80)

        text_branch_type = str(g.get("text_branch_type", "semantic"))
        is_vib_group = text_branch_type == "vib_edl"
        group_cfg_text = dict(cfg_text)
        if not is_vib_group:
            for k in [
                "vib_latent_dim",
                "vib_warmup_epochs",
                "vib_anneal_epochs",
                "vib_kl_weight",
                "vib_logit_l2",
                "vib_dir_kl",
                "vib_mislead",
                "vib_avuc",
                "vib_scale_max",
            ]:
                group_cfg_text.pop(k, None)

        model_kwargs = {
            "in_dim": in_dim,
            "hid_dim": hid_dim,
            "num_classes": num_classes,
            "dropout": dropout,
            "text_branch_type": text_branch_type,
        }
        if is_vib_group:
            model_kwargs["vib_latent_dim"] = int(group_cfg_text.get("vib_latent_dim", hid_dim))
        model = TextOnlyClassifier(**model_kwargs).to(device)

        trainer_kwargs = {
            "q_final": q_final,
            "labels": labels,
            "data_dict": data_dict,
            "model": model,
            "device": device,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "cfg": {"text": group_cfg_text},
            "ckpt_filepath": str(ckpt_path),
            "struct_feats": struct_feats,
            "loss_mode": g["loss_mode"],
            "lambda_u": lambda_u,
            "eval_prob_source": g["eval_prob_source"],
            "early_stop_patience": early_stop_patience,
            "min_delta": min_delta,
            "lambda_u_warmup_epochs": lambda_u_warmup_epochs,
            "scheduler_patience": scheduler_patience,
            "scheduler_factor": scheduler_factor,
            "min_lr": min_lr,
            "checkpoint_metric": checkpoint_metric,
            "detach_semantics_for_u": detach_semantics_for_u,
            "valid_split_seed": valid_split_seed,
        }
        if is_vib_group:
            trainer_kwargs.update(
                {
                    "vib_warmup_epochs": vib_warmup_epochs,
                    "vib_anneal_epochs": vib_anneal_epochs,
                    "vib_kl_weight": vib_kl_weight,
                    "vib_logit_l2": vib_logit_l2,
                    "vib_dir_kl": vib_dir_kl,
                    "vib_mislead": vib_mislead,
                    "vib_avuc": vib_avuc,
                    "vib_scale_max": vib_scale_max,
                }
            )
        trainer = TextTrainer(**trainer_kwargs)
        res = trainer.train(
            run_posthoc=False,
            save_per_node=True,
            calibrators=calibrators,
            artifact_tag=f"seed{seed}",
        )
        outputs = {"raw": res.get("test_raw", {})}
        outputs.update(res.get("test_calibrated", {}))
        val_outputs = {"raw": res.get("val_raw", {})}
        val_outputs.update(res.get("val_calibrated", {}))
        calibrator_params = res.get("calibrator_params", {})

        _append_group_result(
            group_spec=g,
            outputs=outputs,
            val_outputs=val_outputs,
            calibrator_params=calibrator_params,
            ckpt_path_now=ckpt_path,
            run_dir=run_dir,
            run_seed=run_seed,
            source_group_id=g["id"],
            is_posthoc_view=False,
        )

        for derived in derived_groups.get(g["id"], []):
            derived_dir = ckpt_root / derived["id"]
            _materialize_view_artifacts(run_dir, derived_dir, seed, list(outputs.keys()))
            derived_summary_path = derived_dir / f"text_metrics_summary_seed{seed}.json"
            with open(derived_summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_group_id": g["id"],
                        "selected_output": _resolve_selected_output(outputs, derived["default_output"]),
                        "val_raw": res.get("val_raw", {}),
                        "test_raw": res.get("test_raw", {}),
                        "val_calibrated": res.get("val_calibrated", {}),
                        "test_calibrated": res.get("test_calibrated", {}),
                        "calibrator_params": calibrator_params,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            _append_group_result(
                group_spec=derived,
                outputs=outputs,
                val_outputs=val_outputs,
                calibrator_params=calibrator_params,
                ckpt_path_now=ckpt_path,
                run_dir=derived_dir,
                run_seed=run_seed,
                source_group_id=g["id"],
                is_posthoc_view=True,
            )

    json_path = ckpt_root / f"text_matrix_seed{seed}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2, ensure_ascii=False)

    full_json_path = ckpt_root / f"text_matrix_seed{seed}_full.json"
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(full_rows, f, indent=2, ensure_ascii=False)

    csv_path = ckpt_root / f"text_matrix_seed{seed}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    print("\nText-only Matrix Summary")
    print("| Method | F1 | Acc | ECE | Brier | NLL | AURC |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in table_rows:
        print(
            f"| {r['method']} | {r['f1']:.4f} | {r['acc']:.4f} | {r['ece']:.4f} | "
            f"{r['brier']:.4f} | {r['nll']:.4f} | {r['aurc']:.4f} |"
        )
    print(f"\nSaved matrix table to {csv_path}")
    print(f"Saved matrix json to  {json_path}")
    print(f"Saved matrix full to  {full_json_path}")

    return table_rows
