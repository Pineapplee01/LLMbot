"""
text_model.py
Text-only classifier wrapper around text heads.
"""

import torch
import torch.nn as nn

from model import TextClasswiseEvidenceHead, TextFinalSemanticHead, TextVIBEDLHead


class TextOnlyClassifier(nn.Module):
    """
    Thin wrapper for standalone text classification/calibration experiments.

    Input:
        q_final: [B, D] final-layer embeddings
    Output:
        alpha_text, logits_text, temperature_text, concentration_text, u_text, prob_text
    """

    def __init__(
        self,
        in_dim=4096,
        hid_dim=256,
        num_classes=2,
        dropout=0.1,
        text_branch_type="semantic",
        vib_latent_dim=256,
    ):
        super().__init__()
        self.text_branch_type = text_branch_type
        if text_branch_type == "vib_edl":
            self.text_branch = TextVIBEDLHead(
                in_dim=in_dim,
                hid_dim=hid_dim,
                latent_dim=vib_latent_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif text_branch_type == "classwise_edl":
            self.text_branch = TextClasswiseEvidenceHead(
                in_dim=in_dim,
                hid_dim=hid_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            self.text_branch = TextFinalSemanticHead(
                in_dim=in_dim,
                hid_dim=hid_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

    def forward(self, q_final: torch.Tensor, cfg_text=None):
        out = self.text_branch(q_final, cfg_text)
        alpha = out["alpha_text"]
        out["prob_text"] = alpha / alpha.sum(dim=1, keepdim=True)
        return out
