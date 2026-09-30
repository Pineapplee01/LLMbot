"""
utils/calibration.py
Semantic post-hoc calibrators (TS, Beta, ATS) and Qwen3 encoding helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


# ── Calibrators ───────────────────────────────────────────────────────────────

def _binary_margin(logits: torch.Tensor) -> torch.Tensor:
    return (logits[:, 1] - logits[:, 0]).view(-1)


def _entropy_binary(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(min=1e-8, max=1.0 - 1e-8)
    return -(p * torch.log(p)).sum(dim=1)


def _probs_from_margin(margin: torch.Tensor) -> torch.Tensor:
    p1 = torch.sigmoid(margin)
    return torch.stack([1.0 - p1, p1], dim=1)


@dataclass
class CalibratorState:
    name: str
    payload: Dict


class ATSCalibrator(nn.Module):
    """Adaptive Temperature Scaling: T_i = Softplus(a*H + b*u + c)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.ones(1))

    def forward(self, margin: torch.Tensor, entropy: torch.Tensor, u_sem: torch.Tensor) -> torch.Tensor:
        t = F.softplus(self.a * entropy + self.b * u_sem + self.c) + 1e-4
        return margin / t


class SemanticCalibrator:
    """Post-hoc calibration for the semantic expert (TS / Beta / ATS)."""

    def __init__(self, mode: str = "none", device: Optional[torch.device] = None) -> None:
        if mode not in {"none", "ts", "beta", "ats"}:
            raise ValueError(f"Unsupported calibration mode: {mode}")
        self.mode = mode
        self.device = device or torch.device("cpu")
        self.state: Optional[CalibratorState] = None

    def fit(self, logits_sem, prob_sem, u_sem, labels, max_iter=200, lr=1e-2) -> CalibratorState:
        logits_sem = logits_sem.detach().float().to(self.device)
        prob_sem = prob_sem.detach().float().to(self.device)
        u_sem = u_sem.detach().float().view(-1).to(self.device)
        labels = labels.detach().long().view(-1).to(self.device)

        if self.mode == "none":
            self.state = CalibratorState("none", {})
            return self.state

        if self.mode == "ts":
            t = nn.Parameter(torch.ones(1, device=self.device))
            opt = torch.optim.LBFGS([t], lr=0.1, max_iter=max_iter)
            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(logits_sem / t.clamp(min=0.05), labels)
                loss.backward()
                return loss
            opt.step(closure)
            self.state = CalibratorState("ts", {"temperature": float(t.clamp(min=0.05).item())})
            return self.state

        if self.mode == "beta":
            m = _binary_margin(logits_sem).detach().cpu().numpy()
            p = np.clip(1.0 / (1.0 + np.exp(-m)), 1e-8, 1 - 1e-8)
            x = np.stack([np.log(p), -np.log(1.0 - p)], axis=1)
            y = labels.detach().cpu().numpy().astype(np.int64)
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=max_iter)
            model.fit(x, y)
            self.state = CalibratorState("beta", {
                "coef": model.coef_.reshape(-1).astype(float).tolist(),
                "intercept": model.intercept_.reshape(-1).astype(float).tolist(),
                "model": model,
            })
            return self.state

        # ATS
        ats = ATSCalibrator().to(self.device)
        opt = torch.optim.AdamW(ats.parameters(), lr=lr, weight_decay=1e-4)
        margin = _binary_margin(logits_sem)
        ent = _entropy_binary(prob_sem)
        for _ in range(int(max_iter)):
            opt.zero_grad()
            ms = ats(margin, ent, u_sem)
            F.cross_entropy(torch.stack([-0.5 * ms, 0.5 * ms], dim=1), labels).backward()
            opt.step()
        self.state = CalibratorState("ats", {
            "a": float(ats.a.item()), "b": float(ats.b.item()), "c": float(ats.c.item()),
        })
        return self.state

    def apply(self, logits_sem, prob_sem, u_sem) -> Dict[str, torch.Tensor]:
        logits_sem = logits_sem.detach().float().to(self.device)
        prob_sem = prob_sem.detach().float().to(self.device)
        u_sem = u_sem.detach().float().view(-1).to(self.device)
        margin = _binary_margin(logits_sem)

        if self.state is None or self.state.name == "none":
            prob_cal = prob_sem
        elif self.state.name == "ts":
            prob_cal = F.softmax(logits_sem / max(float(self.state.payload["temperature"]), 0.05), dim=1)
        elif self.state.name == "beta":
            model = self.state.payload.get("model")
            m = margin.detach().cpu().numpy()
            p = np.clip(1.0 / (1.0 + np.exp(-m)), 1e-8, 1 - 1e-8)
            x = np.stack([np.log(p), -np.log(1.0 - p)], axis=1)
            p1 = torch.from_numpy(model.predict_proba(x)[:, 1].astype(np.float32)).to(self.device)
            prob_cal = torch.stack([1.0 - p1, p1], dim=1)
        elif self.state.name == "ats":
            a, b, c = self.state.payload["a"], self.state.payload["b"], self.state.payload["c"]
            t = F.softplus(a * _entropy_binary(prob_sem) + b * u_sem + c) + 1e-4
            prob_cal = _probs_from_margin(margin / t)
        else:
            raise ValueError(f"Unknown calibrator state: {self.state.name}")

        pred = prob_cal.argmax(dim=1)
        q_sem = prob_cal.gather(1, pred.unsqueeze(1)).squeeze(1)
        return {"prob_sem_cal": prob_cal, "q_sem": q_sem}


# ── Qwen3 encoding helpers ────────────────────────────────────────────────────

FIXED_INSTRUCTION = "Represent this social media user for bot detection."
INSTRUCTION_PREFIX = "Instruct: "
INPUT_PREFIX = "\nInput: "


def build_account_text(text: str, max_length: int = 2048, mode: str = "clean", seed: int = 42, node_id: int = 0) -> str:
    if not isinstance(text, str):
        return ""
    for old, new in {" </s> ": "\n", "METADATA:": "## Profile:\n", "DESCRIPTION:": "\n## Bio:\n", "TWEET:": "\n## Tweets:\n"}.items():
        text = text.replace(old, new)
    words = text.strip().split()
    if mode == "perturb" and len(words) >= 3:
        import numpy as np
        rng = np.random.default_rng(seed + node_id)
        words = [w for w in words if rng.random() > 0.2] or words[:1]
    return " ".join(words[: int((max_length - 50) * 0.75)])


def build_instructioned_text(text: str, instruction_mode: str = "on", prompt_mode: str = "clean", max_length: int = 2048, seed: int = 42, node_id: int = 0) -> str:
    if prompt_mode == "raw_prompt":
        return text if isinstance(text, str) else ""
    cleaned = build_account_text(text, max_length=max_length, mode=prompt_mode, seed=seed, node_id=node_id)
    instruction = FIXED_INSTRUCTION if instruction_mode == "on" else ""
    if instruction:
        return f"{INSTRUCTION_PREFIX}{instruction}{INPUT_PREFIX}{cleaned}"
    return cleaned


def encode_accounts(generator, texts, node_ids, mode: str = "clean"):
    all_emb = None
    for i in range(0, len(texts), generator.batch_size):
        emb_dict = generator.encode_batch(texts[i:i + generator.batch_size], node_ids[i:i + generator.batch_size], mode=mode)
        if all_emb is None:
            all_emb = {k: [] for k in emb_dict}
        for k, v in emb_dict.items():
            all_emb[k].append(v)
    return {k: torch.cat(v, dim=0) for k, v in all_emb.items()} if all_emb else {}
